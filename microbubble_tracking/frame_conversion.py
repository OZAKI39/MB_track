"""送帧端固定转换空间；借出期间禁止覆盖，复制到等待区后才能归还。

仅用于当前 WSL 软件回放的 yuv420p、未指定色彩空间/范围输入。
使用 PyAV 已加载的 FFmpeg，不猜测 AVFrame 私有结构或加载另一套系统库。
"""

import ctypes as ct
import sys
import threading

import av
import av.video.reformatter
import cv2
import numpy as np


class ReusableConversion:
    """单个发送线程持有一组彩色/灰度空间，最多借出一张正在发送的画面。

    convert_bgr() 借出空间，to_gray() 在同一次借用内完成灰度转换。
    调用者必须等所有同步读取/复制完成再 release()，之后旧视图不可再使用。
    异步消费者不能保存这里的视图；应像 LatestFrame 一样先取得自己的副本。
    """

    def __init__(self, frame, threads=None):
        self._context = None
        self.in_use = False
        self._owner_thread = threading.get_ident()
        self._signature = self._frame_signature(frame)
        if self._signature[2:] != ("yuv420p", 2, 0):
            raise ValueError(
                "Reusable conversion currently requires yuv420p with unspecified "
                "colorspace/range; use --conversion-memory allocate for other inputs"
            )
        if not sys.platform.startswith("linux"):
            raise RuntimeError(
                "Reusable conversion is currently validated on Linux/WSL only"
            )
        # 从当前 PyAV 扩展的依赖中解析公开函数，避免混用不同版本 FFmpeg。
        self._library = ct.CDLL(av.video.reformatter.__file__)
        pointer, integer = ct.c_void_p, ct.c_int
        self._alloc = self._bind("sws_alloc_context", pointer, [])
        self._free = self._bind("sws_freeContext", None, [pointer])
        self._option = self._bind(
            "av_opt_set_int", integer, [pointer, ct.c_char_p, ct.c_int64, integer]
        )
        self._init = self._bind(
            "sws_init_context", integer, [pointer, pointer, pointer]
        )
        self._pixel_format = self._bind("av_get_pix_fmt", integer, [ct.c_char_p])
        self._scale = self._bind(
            "sws_scale",
            integer,
            [
                pointer,
                ct.POINTER(pointer),
                ct.POINTER(integer),
                integer,
                integer,
                ct.POINTER(pointer),
                ct.POINTER(integer),
            ],
        )
        version = self._bind("swscale_version", ct.c_uint, [])()
        expected = av.library_versions["libswscale"]
        if version != (expected[0] << 16 | expected[1] << 8 | expected[2]):
            raise RuntimeError("FFmpeg conversion library does not match the decoder")

        self.width, self.height = frame.width, frame.height
        # PyAV 分配符合 FFmpeg 对齐/填充要求的平面；numpy 只建立视图，不再复制。
        self._target = av.VideoFrame(self.width, self.height, "bgr24")
        plane = self._target.planes[0]
        self.bgr = np.ndarray(
            (self.height, self.width, 3),
            np.uint8,
            buffer=plane,
            strides=(plane.line_size, 3, 1),
        )
        np.frombuffer(plane, np.uint8).fill(0)
        self.gray = np.empty((self.height, self.width), np.uint8)
        self.gray.fill(0)
        self._source_pointers = (pointer * 4)()
        self._source_strides = (integer * 4)()
        self._target_pointers = (pointer * 4)(plane.buffer_ptr, None, None, None)
        self._target_strides = (integer * 4)(plane.line_size, 0, 0, 0)
        self._context = self._alloc()
        if not self._context:
            raise MemoryError("Cannot allocate FFmpeg conversion context")
        try:
            options = {
                "srcw": self.width,
                "srch": self.height,
                "dstw": self.width,
                "dsth": self.height,
                "src_format": self._pixel_format(b"yuv420p"),
                "dst_format": self._pixel_format(b"bgr24"),
                "sws_flags": 2,  # SWS_BILINEAR，与原 PyAV 调用一致。
                "threads": 0 if threads is None else threads,
            }
            for name, value in options.items():
                if self._option(self._context, name.encode("ascii"), value, 0) < 0:
                    raise RuntimeError(f"Cannot configure conversion option: {name}")
            if self._init(self._context, None, None) < 0:
                raise RuntimeError("Cannot initialize reusable conversion")
            # 不把运行库兼容性当作假设：每次启动先核对真实首帧，失败就停止。
            kwargs = {} if threads is None else {"threads": threads}
            expected_bgr = frame.to_ndarray(format="bgr24", **kwargs)
            actual = self.convert_bgr(frame)
            if not np.array_equal(actual, expected_bgr):
                raise ValueError(
                    "Reusable conversion differs from PyAV; playback not started"
                )
            self.release()
        except BaseException:
            self.in_use = False
            self.close()
            raise
        self.metadata = {
            "mode": "reuse",
            "conversion_buffers": 1,
            "bgr_bytes": plane.buffer_size,
            "gray_bytes": self.gray.nbytes,
            "bgr_address": self.bgr.ctypes.data,
            "gray_address": self.gray.ctypes.data,
            "swscale_version": list(expected),
            "library": av.video.reformatter.__file__,
            "startup_pixel_check": "exact",
            "release_after": "LatestFrame.put copy",
        }

    def _bind(self, name, result, arguments):
        function = getattr(self._library, name)
        function.restype = result
        function.argtypes = arguments
        return function

    @staticmethod
    def _frame_signature(frame):
        return (
            frame.width,
            frame.height,
            frame.format.name,
            frame.colorspace,
            frame.color_range,
        )

    def _check_owner(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Conversion buffers belong to a different thread")

    def convert_bgr(self, frame):
        """写入空闲的固定彩色空间；只借出视图，不把所有权交给调用者。"""
        self._check_owner()
        if not self._context:
            raise RuntimeError("Conversion buffers are closed")
        if self.in_use:
            raise RuntimeError(
                "Previous frame is still in use; release it before conversion"
            )
        if self._frame_signature(frame) != self._signature:
            raise ValueError(
                "Frame size, format or color metadata changed during playback"
            )
        planes = frame.planes
        if len(planes) != 3:
            raise ValueError("Expected three CPU-readable YUV planes")
        for index, plane in enumerate(planes):
            if (
                plane.line_size < plane.width
                or plane.buffer_size < plane.line_size * plane.height
            ):
                raise ValueError("Invalid source plane storage")
            self._source_pointers[index] = plane.buffer_ptr
            self._source_strides[index] = plane.line_size
        # frame/planes 在同步调用结束前保持存活；函数返回后不再读取这些指针。
        rows = self._scale(
            self._context,
            self._source_pointers,
            self._source_strides,
            0,
            self.height,
            self._target_pointers,
            self._target_strides,
        )
        if rows != self.height:
            raise RuntimeError(f"Incomplete conversion: {rows}/{self.height} rows")
        self.in_use = True
        return self.bgr

    def to_gray(self):
        """在已借出的同一帧内，写入固定灰度空间。"""
        self._check_owner()
        if not self.in_use:
            raise RuntimeError("No frame has been acquired")
        result = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY, dst=self.gray)
        if result is not self.gray:
            raise RuntimeError("OpenCV did not use the supplied gray buffer")
        return self.gray

    def release(self):
        """所有同步读取者完成后归还；归还后旧视图不能再交给其他消费者。"""
        self._check_owner()
        if not self.in_use:
            raise RuntimeError("No frame to release")
        self.in_use = False

    def close(self):
        """结束转换；禁止在仍有借用时释放转换器。"""
        self._check_owner()
        if self.in_use:
            raise RuntimeError("Cannot close conversion while a frame is in use")
        if self._context:
            self._free(self._context)
            self._context = None
