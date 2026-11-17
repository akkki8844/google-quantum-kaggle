"""Push a local folder as a private Kaggle script kernel, check its status, fetch its output.

Needs the official `kaggle` CLI and your API token at ~/.kaggle/kaggle.json (you create it on
kaggle.com -> Settings -> API -> Create New Token). This script reads only the `username`
field from that file and never prints or copies the key.

Usage (from the project root):
    .venv/Scripts/python.exe tools/kpush.py push   kaggle_nb/p0_reference [--gpu] [--internet] [--dataset owner/slug ...]
    .venv/Scripts/python.exe tools/kpush.py status kaggle_nb/p0_reference
    .venv/Scripts/python.exe tools/kpush.py output kaggle_nb/p0_reference

The folder must hold exactly one .py file (the kernel code). Kernels are always private.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

COMPETITION = "google-verifiable-quantum-circuit-compilation"


def username() -> str:
    """Username as the CLI resolves it (works for kaggle.json and the newer access_token file)."""
    cache = pathlib.Path(__file__).with_name(".kaggle_username")
    if cache.is_file() and cache.read_text().strip():
        return cache.read_text().strip()
    for _ in range(3):  # `config view` is occasionally flaky; retry before giving up
        out = subprocess.run([kaggle_cli(), "config", "view"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.strip().startswith("- username:"):
                name = line.split(":", 1)[1].strip()
                if name and name != "None":
                    cache.write_text(name)
                    return name
    sys.exit("Kaggle CLI has no username; check ~/.kaggle/access_token or kaggle.json.")


def kaggle_cli() -> str:
    exe = shutil.which("kaggle") or str(pathlib.Path(sys.executable).parent / "kaggle")
    return exe


def kernel_id(folder: pathlib.Path) -> str:
    return f"{username()}/qcc-{folder.name.replace('_', '-')}"


def push(folder: pathlib.Path, gpu: bool, internet: bool, datasets: list[str],
         kernels: list[str] | None = None) -> None:
    code = [p for p in folder.glob("*.py")]
    if len(code) != 1:
        sys.exit(f"{folder} must contain exactly one .py file, found {len(code)}")
    meta = {
        "id": kernel_id(folder),
        "title": f"qcc {folder.name.replace('_', ' ')}",
        "code_file": code[0].name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": internet,
        "dataset_sources": datasets,
        "competition_sources": [COMPETITION],
        "kernel_sources": kernels or [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    cmd = [kaggle_cli(), "kernels", "push", "-p", str(folder)]
    if gpu:
        # Name the accelerator explicitly: a bare enable_gpu push sat QUEUED for over an hour.
        cmd += ["--accelerator", "NvidiaTeslaT4"]
    subprocess.run(cmd, check=True)


def status(folder: pathlib.Path) -> None:
    subprocess.run([kaggle_cli(), "kernels", "status", kernel_id(folder)], check=True)


def output(folder: pathlib.Path) -> None:
    out = folder / "output"
    out.mkdir(exist_ok=True)
    subprocess.run([kaggle_cli(), "kernels", "output", kernel_id(folder), "-p", str(out)], check=True)
    print(f"saved to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "status", "output"])
    ap.add_argument("folder", type=pathlib.Path)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--internet", action="store_true")
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--kernel", action="append", default=[], help="attach another notebook's output")
    a = ap.parse_args()
    if a.action == "push":
        push(a.folder, a.gpu, a.internet, a.dataset, a.kernel)
    elif a.action == "status":
        status(a.folder)
    else:
        output(a.folder)


if __name__ == "__main__":
    main()
