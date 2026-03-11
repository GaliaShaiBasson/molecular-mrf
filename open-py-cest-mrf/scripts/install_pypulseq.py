import os
import shutil
import stat
import subprocess
import sys


def remove_readonly(func, path, exc_info):
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work_dir = os.path.join(repo_root, "cest_mrf")
    os.chdir(work_dir)

    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "pypulseq"])

    if shutil.which("git") is None:
        raise RuntimeError("Git is not installed")

    custom_write_seq = os.path.join(work_dir, "write_seq.py")
    if not os.path.exists(custom_write_seq):
        raise RuntimeError("Custom write_seq.py not found")

    pypulseq_dir = os.path.join(work_dir, "pypulseq")
    if os.path.exists(pypulseq_dir):
        shutil.rmtree(pypulseq_dir, onerror=remove_readonly)

    subprocess.check_call([
        "git", "clone", "--branch", "master",
        "https://github.com/imr-framework/pypulseq"
    ])

    subprocess.check_call(
        ["git", "reset", "--hard", "cc9ccfb"],
        cwd=pypulseq_dir
    )

    target_file = os.path.join(
        pypulseq_dir, "pypulseq", "Sequence", "write_seq.py"
    )
    if os.path.exists(target_file):
        os.remove(target_file)
    shutil.copy(custom_write_seq, target_file)

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."],
        cwd=pypulseq_dir
    )


if __name__ == "__main__":
    main()
