import json
import platform
import subprocess
from pathlib import Path

from .common import dump


def snapshot(root):
    path=root/"hardware.json"
    if path.exists():
        return
    data={"cpu":platform.processor(),"os":platform.platform(),"graphics":[]}
    if platform.system()=="Windows":
        cmd=("$p=Get-CimInstance Win32_Processor | Select-Object -First 1; "
             "$v=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion); "
             "@{cpu=$p.Name;graphics=$v} | ConvertTo-Json -Compress")
        cap=subprocess.run(["powershell","-NoProfile","-NonInteractive","-Command",cmd],
                           capture_output=True,text=True,check=True)
        data.update(json.loads(cap.stdout.strip()))
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                data["cpu"]=line.split(":",1)[1].strip()
                break
    dump(path,data)
