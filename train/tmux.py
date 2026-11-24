"""Start a training command in tmux and print its progress query."""
import os
import shlex
import subprocess


def launch_tmux(session, root, run, worker, host):
    command = ["tmux", "new-session", "-d", "-s", session, "-c", str(root)]
    for key in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                "MKL_THREADING_LAYER", "TOKENIZERS_PARALLELISM", "WANDB_MODE", "WANDB_DIR"):
        if key in os.environ:
            command.extend(["-e", key + "=" + os.environ[key]])
    log_path = run / "run.log"
    command.append("exec " + shlex.join(worker) + " > " + shlex.quote(str(log_path)) + " 2>&1")
    subprocess.run(command, check=True)
    print(f"tmux: {session}\noutput: {run}")
    query = shlex.join(["tail", "-n", "20", str(log_path)])
    print(shlex.join(["ssh", host, query]) if host else query)
