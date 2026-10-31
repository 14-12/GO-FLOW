# Ends any co-simulation still running (python -m cosim.run or python -m cosim.freefly, with its SUMO and the
# viewer's helper processes): a run that failed or hung can stay alive holding SUMO's port and the CARLA
# connection. Simulations of other scripts are left alone.
$procs = Get-CimInstance Win32_Process
$runs = $procs | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'cosim\.(run|freefly)' }
$ids = @($runs.ProcessId)
$kids = $procs | Where-Object { $ids -contains $_.ParentProcessId }
foreach ($p in @($kids) + @($runs)) {
  if ($p) { "stopping {0} pid={1}" -f $p.Name, $p.ProcessId; Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
}
