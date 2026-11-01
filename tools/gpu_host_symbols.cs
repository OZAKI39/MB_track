// Resolve saved ETW addresses using matching Microsoft public symbols; never attach to a process.
using System;
using System.IO;
using System.Diagnostics;
using System.Runtime.InteropServices;

public static class GpuHostSymbols
{
    const string DbgHelp = @"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\dbghelp.dll";
    [DllImport(DbgHelp, SetLastError=true, CharSet=CharSet.Unicode)]
    static extern bool SymInitializeW(IntPtr process, string path, bool invade);
    [DllImport(DbgHelp, SetLastError=true, CharSet=CharSet.Unicode)]
    static extern ulong SymLoadModuleExW(IntPtr process, IntPtr file, string image, string module,
        ulong address, uint size, IntPtr data, uint flags);
    [DllImport(DbgHelp, SetLastError=true)]
    static extern bool SymFromAddr(IntPtr process, ulong address, out ulong displacement, IntPtr symbol);
    [DllImport(DbgHelp)] static extern uint SymSetOptions(uint options);
    [DllImport(DbgHelp)] static extern bool SymCleanup(IntPtr process);

    [StructLayout(LayoutKind.Sequential)]
    struct SymbolInfo
    {
        public uint SizeOfStruct, TypeIndex;
        public ulong Reserved1, Reserved2;
        public uint Index, Size;
        public ulong ModBase;
        public uint Flags;
        public ulong Value, Address;
        public uint Register, Scope, Tag, NameLen, MaxNameLen;
        public byte Name;
    }

    public static void Resolve(string modules, string addresses, string cache, string output)
    {
        if (File.Exists(output)) throw new IOException("Output already exists");
        Directory.CreateDirectory(cache);
        using (var self = Process.GetCurrentProcess())
        using (var writer = new StreamWriter(output))
        {
            IntPtr process = self.Handle;
            SymSetOptions(0x2 | 0x4 | 0x200 | 0x80000); // Undecorate, deferred, fail-critical-errors, no prompts.
            if (!SymInitializeW(process, "srv*" + cache + "*https://msdl.microsoft.com/download/symbols", false))
                throw new InvalidOperationException("SymInitialize: " + Marshal.GetLastWin32Error());
            try
            {
                foreach (string line in File.ReadLines(modules))
                {
                    string[] parts = line.Split('\t');
                    ulong address = Convert.ToUInt64(parts[0], 16);
                    uint size = Convert.ToUInt32(parts[1], 16);
                    ulong loaded = SymLoadModuleExW(process, IntPtr.Zero, parts[2], null,
                        address, size, IntPtr.Zero, 0);
                    Console.WriteLine("Module " + parts[2] + " loaded=" + loaded.ToString("x")
                        + " error=" + (loaded == 0 ? Marshal.GetLastWin32Error() : 0));
                }
                int sizeOfInfo = Marshal.SizeOf(typeof(SymbolInfo));
                int nameOffset = (int)Marshal.OffsetOf(typeof(SymbolInfo), "Name");
                IntPtr buffer = Marshal.AllocHGlobal(sizeOfInfo + 2048);
                try
                {
                    foreach (string line in File.ReadLines(addresses))
                    {
                        ulong address = Convert.ToUInt64(line, 16), displacement;
                        var info = new SymbolInfo { SizeOfStruct=(uint)sizeOfInfo, MaxNameLen=2048 };
                        Marshal.StructureToPtr(info, buffer, false);
                        if (SymFromAddr(process, address, out displacement, buffer))
                        {
                            info = (SymbolInfo)Marshal.PtrToStructure(buffer, typeof(SymbolInfo));
                            string name = Marshal.PtrToStringAnsi(IntPtr.Add(buffer, nameOffset), (int)info.NameLen);
                            writer.WriteLine(address.ToString("x") + "\t" + name + "\t" + displacement);
                        }
                        else writer.WriteLine(address.ToString("x") + "\tUNRESOLVED\t" + Marshal.GetLastWin32Error());
                        writer.Flush();
                    }
                }
                finally { Marshal.FreeHGlobal(buffer); }
            }
            finally { SymCleanup(process); }
        }
    }
}
