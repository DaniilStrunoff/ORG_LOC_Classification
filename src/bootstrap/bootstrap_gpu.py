import os, re, subprocess, sys
from types import SimpleNamespace
from typing import List

TORCH_VERSION = os.getenv("BOOTSTRAP_TORCH_VERSION", "2.9.1")
BNB_VERSION = os.getenv("BOOTSTRAP_BNB_VERSION", "0.43.1")
ACCELERATE_VERSION = os.getenv("BOOTSTRAP_ACCELERATE_VERSION", "0.34.0")
UNSLOTH_VERSION = os.getenv("BOOTSTRAP_UNSLOTH_VERSION", "2024.7.0")
PEFT_VERSION = os.getenv("BOOTSTRAP_PEFT_VERSION", "0.11.1")


def run(cmd: List[str]):
    try:
        return subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError as e:
        return SimpleNamespace(returncode=127, stdout="", stderr=str(e))


def pip_install(args: List[str]):
    subprocess.check_call([sys.executable, "-m", "pip"] + args)


def have(mod: str):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def driver_cuda_mm():
    r = run(["nvidia-smi"])
    if r.returncode != 0:
        return None
    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", r.stdout)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def pick_torch_index():
    mm = driver_cuda_mm()
    if mm and mm.startswith("12."):
        return "https://download.pytorch.org/whl/cu126"
    elif mm and mm.startswith("13."):
        return "https://download.pytorch.org/whl/cu126"
    else:
        return "https://download.pytorch.org/whl/cpu"


def ensure_torch():
    if have("torch"):
        sys.stderr.write("[bootstrap] torch already installed; skipping\n")
        return
    index = pick_torch_index()
    try:
        pip_install(
            [
                "install",
                "--no-cache-dir",
                f"torch=={TORCH_VERSION}",
                "--index-url",
                index,
            ]
        )
        sys.stderr.write(f"[bootstrap] installed torch=={TORCH_VERSION} from {index}\n")
    except Exception as e:
        sys.stderr.write(f"[bootstrap] torch install failed: {e}\n")


def ensure_accelerate():
    if have("accelerate"):
        sys.stderr.write("[bootstrap] accelerate already installed; skipping\n")
        return
    try:
        pip_install(
            [
                "install",
                "--no-cache-dir",
                f"accelerate>={ACCELERATE_VERSION}",
            ]
        )
        sys.stderr.write(f"[bootstrap] installed accelerate>={ACCELERATE_VERSION}\n")
    except Exception as e:
        sys.stderr.write(f"[bootstrap] accelerate install failed: {e}\n")


def ensure_peft():
    if have("peft"):
        sys.stderr.write("[bootstrap] peft already installed; skipping\n")
        return
    try:
        pip_install(
            [
                "install",
                "--no-cache-dir",
                "--no-deps",
                f"peft=={PEFT_VERSION}",
            ]
        )
        sys.stderr.write(f"[bootstrap] installed peft=={PEFT_VERSION}\n")
    except Exception as e:
        sys.stderr.write(f"[bootstrap] peft install failed: {e}\n")


def main():
    ensure_torch()
    ensure_accelerate()
    ensure_peft()


if __name__ == "__main__":
    main()
