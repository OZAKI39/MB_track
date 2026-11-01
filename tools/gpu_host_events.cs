// Read existing ETL with Microsoft's TraceEvent library. No recording or settings changes.
using System;
using System.IO;
using System.Text;
using System.Globalization;
using Microsoft.Diagnostics.Tracing;

public static class GpuHostEvents
{
    public static void Read(string input, string output, double[] starts, double[] ends)
    {
        ReadSelected(input, output, starts, ends, false);
    }

    public static void ReadOrigins(string input, string output)
    {
        ReadSelected(input, output, new double[0], new double[0], true);
    }

    static void ReadSelected(string input, string output, double[] starts, double[] ends, bool origins)
    {
        if (File.Exists(output)) throw new IOException("Output already exists");
        long selected = 0, errors = 0;
        using (var writer = new StreamWriter(output, false, new UTF8Encoding(false)))
        using (var source = new ETWTraceEventSource(input))
        {
            Console.WriteLine("Session UTC: " + source.SessionStartTime.ToUniversalTime().ToString("o"));
            Action<TraceEvent> emit = delegate(TraceEvent item)
            {
                double stamp = item.TimeStampRelativeMSec;
                // Always retain the rare ACPI notifications, including ones away from selected slow frames.
                bool keep = item.ProviderName == "Microsoft-Windows-DxgKrnl"
                    && (item.EventName == "ACPIEvent" || item.EventName.StartsWith("DdiNotifyAcpiEvent/"));
                if (origins)
                {
                    string provider = item.ProviderName ?? "";
                    keep |= provider.IndexOf("Acpi", StringComparison.OrdinalIgnoreCase) >= 0
                        || provider == "Microsoft-Windows-WMI-Activity"
                        || item.ProviderGuid == new Guid("dab01d4d-2d48-477d-b1c3-daad0ce6f06b");
                    if (provider == "Microsoft-Windows-DxgKrnl")
                    {
                        keep |= item.EventName.StartsWith("TraceAdapterLock");
                        if (item.EventName.StartsWith("Profiler/"))
                        {
                            int field = Array.IndexOf(item.PayloadNames, "Function");
                            string function = field < 0 ? "" : item.PayloadString(field);
                            keep |= function == "1009" || function == "DxgkLock2"
                                || function == "DxgkUnlock2" || function == "DxgkSubmitCommandToHwQueue"
                                || function == "DxgkSubmitCommand";
                        }
                    }
                }
                for (int i = 0; i < starts.Length; ++i)
                    if (stamp >= starts[i] && stamp <= ends[i]) { keep = true; break; }
                if (!keep) return;
                writer.Write(stamp.ToString("F6", CultureInfo.InvariantCulture));
                writer.Write("\t" + item.ProviderName + "\t" + item.EventName);
                writer.Write("\t" + item.ProcessID + "\t" + item.ThreadID);
                for (int i = 0; i < item.PayloadNames.Length; ++i)
                {
                    string value;
                    try {
                        value = item.PayloadString(i);
                        if (value.Contains("EXCEPTION_DURING_VALUE_LOOKUP")) ++errors;
                    }
                    catch (Exception ex) { value = "DECODE_ERROR:" + ex.GetType().Name; ++errors; }
                    writer.Write("\t" + item.PayloadNames[i] + "=" + value.Replace("\t", " ").Replace("\n", " ").Replace("\r", " "));
                }
                writer.WriteLine();
                ++selected;
            };
            source.Kernel.All += emit;
            source.Dynamic.All += emit;
            source.Process();
            Console.WriteLine("Selected events: " + selected);
            Console.WriteLine("Field decode errors: " + errors);
            Console.WriteLine("Events lost: " + source.EventsLost);
        }
    }
}
