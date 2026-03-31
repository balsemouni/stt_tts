import os
import sys
import subprocess
import time
import psutil
import torch

# ======================================================
# 🔐 CRITICAL GPU PROCESSES (DO NOT TOUCH)
# ======================================================

CRITICAL_KEYWORDS = {
    "nvidia",
    "nvdisplay",
    "nvcontainer",
    "system",
    "wininit",
    "winlogon",
    "dwm",
    "csrss",
    "services",
    "lsass",
    "smss",
    "svchost",
    "explorer",
}

# ======================================================
# 🧠 HELPERS
# ======================================================

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode()
    except:
        return ""

def is_critical(name: str, pid: int) -> bool:
    name = name.lower()
    if pid == os.getpid():
        return True
    return any(k in name for k in CRITICAL_KEYWORDS)

def get_gpu_processes():
    out = run(
        "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory "
        "--format=csv,noheader,nounits"
    )
    procs = []
    for line in out.splitlines():
        try:
            parts = line.split(",")
            if len(parts) >= 3:
                pid = int(parts[0].strip())
                name = parts[1].strip()
                mem = int(parts[2].strip())
                procs.append({"pid": pid, "name": name, "mem": mem})
        except Exception as e:
            print(f"⚠️ Warning parsing line: {line} - {e}")
    return procs


def free_gpu_smart(min_mem_mb=100):
    print("\n🧹 SAFE GPU MEMORY RECLAIMER")
    print("=" * 60)

    gpu_procs = get_gpu_processes()
    if not gpu_procs:
        print("✅ No GPU processes found - GPU is clean")
        return 0

    critical_procs = []
    small_procs = []
    movable_procs = []

    for proc in gpu_procs:
        if is_critical(proc["name"], proc["pid"]):
            critical_procs.append(proc)
        elif proc["mem"] < min_mem_mb:
            small_procs.append(proc)
        else:
            movable_procs.append(proc)

    print(f"\n📊 Found {len(gpu_procs)} GPU processes:")
    print(f"   🔒 Critical (protected): {len(critical_procs)}")
    print(f"   🟡 Small usage (<{min_mem_mb}MB): {len(small_procs)}")
    print(f"   🎯 Can be moved: {len(movable_procs)}")

    if not movable_procs:
        print("\n✅ No processes to move")
        return 0

    return 0


def force_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if not torch.cuda.is_available():
        print("❌ CUDA NOT AVAILABLE — EXITING")
        sys.exit(1)

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    print("\n🚀 GPU FORCED (cuda:0)")
    print(f"🧠 GPU: {torch.cuda.get_device_name(0)}")

    free_mem = torch.cuda.mem_get_info()[0] // 1024**2
    total_mem = torch.cuda.mem_get_info()[1] // 1024**2

    print(f"📦 GPU Memory:")
    print(f"   Free: {free_mem}MB")
    print(f"   Total: {total_mem}MB")
    print(f"   Used: {total_mem - free_mem}MB")

    return torch.device("cuda:0")


def get_gpu_memory_info():
    if not torch.cuda.is_available():
        return None

    free_mem = torch.cuda.mem_get_info()[0] // 1024**2
    total_mem = torch.cuda.mem_get_info()[1] // 1024**2
    used_mem = total_mem - free_mem

    return {
        'free_mb': free_mem,
        'total_mb': total_mem,
        'used_mb': used_mem,
        'utilization': (used_mem / total_mem * 100) if total_mem > 0 else 0
    }


def cleanup_gpu_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
