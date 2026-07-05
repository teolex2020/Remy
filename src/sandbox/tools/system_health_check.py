import os
import platform
import subprocess

TOOL_NAME = "system_health_check"
TOOL_DESCRIPTION = "Retrieves system metrics like CPU load and memory usage (Windows/Linux)."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {},
    "required": []
}

def execute():
    """
    Retrieves system metrics like CPU load and memory usage.
    """
    system = platform.system()
    metrics = {"os": system}
    
    try:
        if system == "Windows":
            # CPU Usage
            cpu_cmd = "wmic cpu get loadpercentage"
            cpu_out = subprocess.check_output(cpu_cmd, shell=True).decode().strip().split('\n')
            if len(cpu_out) > 1:
                metrics["cpu_load_percent"] = cpu_out[1].strip()
            
            # Memory Usage
            mem_cmd = "wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value"
            mem_out = subprocess.check_output(mem_cmd, shell=True).decode().strip().split('\n')
            for line in mem_out:
                if "=" in line:
                    key, val = line.split('=')
                    if "FreePhysicalMemory" in key:
                        metrics["free_mem_kb"] = val.strip()
                    if "TotalVisibleMemorySize" in key:
                        metrics["total_mem_kb"] = val.strip()
                    
            if "free_mem_kb" in metrics and "total_mem_kb" in metrics:
                free = int(metrics["free_mem_kb"])
                total = int(metrics["total_mem_kb"])
                used = total - free
                metrics["mem_usage_percent"] = round((used / total) * 100, 2)
                
        elif system == "Linux":
            load1, load5, load15 = os.getloadavg()
            metrics["load_avg"] = [load1, load5, load15]
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if "MemTotal" in line:
                            metrics["total_mem_kb"] = line.split()[1]
                        if "MemAvailable" in line:
                            metrics["available_mem_kb"] = line.split()[1]
        
        return str(metrics)
    except Exception as e:
        return f"Error collecting metrics: {str(e)}"

def test_execute():
    res = execute()
    print(res)
    assert "os" in res
