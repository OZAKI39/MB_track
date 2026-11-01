// Read-only CUDA copy observer for this diagnostic. Never changes a copy or its stream.
// CUPTI callbacks record directions and byte counts; unknown APIs stay explicitly unknown.
#include <cstdio>
#include <cstring>
#include <mutex>
#include <cupti.h>

static CUpti_SubscriberHandle subscriber;
static FILE* output = nullptr;
static std::mutex output_mutex;
static bool active = false;

static void CUPTIAPI observe(void*, CUpti_CallbackDomain domain,
                            CUpti_CallbackId, const void* data) {
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    const auto* info = static_cast<const CUpti_CallbackData*>(data);
    if (info->callbackSite != CUPTI_API_ENTER) return;
    const char* name = info->functionName;
    if (std::strncmp(name, "cuMemcpy", 8) != 0) return;
    size_t bytes = 0;
    int src = 0, dst = 0;
    bool known = true;
    if (std::strncmp(name, "cuMemcpy2D", 10) == 0) {
        // Both sync and async v2 structs start with the same pCopy member.
        const auto* p = static_cast<const cuMemcpy2DAsync_v2_params*>(info->functionParams)->pCopy;
        src = p->srcMemoryType; dst = p->dstMemoryType;
        bytes = p->WidthInBytes * p->Height;
    } else if (std::strncmp(name, "cuMemcpy3D", 10) == 0 && std::strstr(name, "Peer") == nullptr) {
        const auto* p = static_cast<const cuMemcpy3DAsync_v2_params*>(info->functionParams)->pCopy;
        src = p->srcMemoryType; dst = p->dstMemoryType;
        bytes = p->WidthInBytes * p->Height * p->Depth;
    } else if (std::strncmp(name, "cuMemcpyHtoD", 12) == 0) {
        const auto* p = static_cast<const cuMemcpyHtoDAsync_v2_params*>(info->functionParams);
        src = CU_MEMORYTYPE_HOST; dst = CU_MEMORYTYPE_DEVICE; bytes = p->ByteCount;
    } else if (std::strncmp(name, "cuMemcpyDtoH", 12) == 0) {
        const auto* p = static_cast<const cuMemcpyDtoHAsync_v2_params*>(info->functionParams);
        src = CU_MEMORYTYPE_DEVICE; dst = CU_MEMORYTYPE_HOST; bytes = p->ByteCount;
    } else if (std::strncmp(name, "cuMemcpyDtoD", 12) == 0) {
        const auto* p = static_cast<const cuMemcpyDtoDAsync_v2_params*>(info->functionParams);
        src = CU_MEMORYTYPE_DEVICE; dst = CU_MEMORYTYPE_DEVICE; bytes = p->ByteCount;
    } else {
        known = false;
    }
    std::lock_guard<std::mutex> lock(output_mutex);
    if (output) std::fprintf(output, "%s,%d,%d,%zu,%d\n", name, src, dst, bytes, known);
}

extern "C" int audit_start(const char* path) {
    if (active) return -1;
    output = std::fopen(path, "wx");
    if (!output) return -2;
    std::fprintf(output, "api,source_type,destination_type,bytes,known\n");
    CUptiResult result = cuptiSubscribe(&subscriber, observe, nullptr);
    if (result == CUPTI_SUCCESS) {
        result = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
        if (result != CUPTI_SUCCESS) cuptiUnsubscribe(subscriber);
    }
    if (result != CUPTI_SUCCESS) {
        const char* reason = nullptr;
        cuptiGetResultString(result, &reason);
        std::fprintf(stderr, "CUPTI observer: %s\n", reason ? reason : "unknown error");
        std::fclose(output); output = nullptr;
        return static_cast<int>(result);
    }
    active = true;
    return 0;
}

extern "C" void audit_stop() {
    if (!active) return;
    cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    cuptiUnsubscribe(subscriber);
    std::lock_guard<std::mutex> lock(output_mutex);
    if (output) std::fclose(output);
    output = nullptr;
    active = false;
}
