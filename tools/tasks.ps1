# Processes that matter for a run (CARLA, Python, SUMO, consoles) with command lines; listening ports; GPU.
Get-Date
$os = Get-CimInstance Win32_OperatingSystem
'RAM total {0:N1} GB, free {1:N1} GB; commit limit {2:N1} GB, commit free {3:N1} GB' -f ($os.TotalVisibleMemorySize/1MB), ($os.FreePhysicalMemory/1MB), ($os.TotalVirtualMemorySize/1MB), ($os.FreeVirtualMemory/1MB)
(Get-Counter '\Memory\Pages/sec','\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue).CounterSamples | ForEach-Object { '{0} = {1:N0}' -f $_.Path, $_.CookedValue }
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'Carla|python|sumo|cmd.exe|nvidia-smi' } |
  ForEach-Object { '{0} pid={1} parent={2} started={3:HH:mm:ss} cpu_s={4:N0} mem_mb={5:N0} cmd={6}' -f $_.Name, $_.ProcessId, $_.ParentProcessId, $_.CreationDate, (($_.KernelModeTime + $_.UserModeTime) / 1e7), ($_.WorkingSetSize / 1MB), $_.CommandLine }
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in 2000, 2001, 8813, 8890, 8891, 8892 } |
  ForEach-Object { 'listen {0} pid={1}' -f $_.LocalPort, $_.OwningProcess }
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv
