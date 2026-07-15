#!/usr/bin/env python3
"""Upload train_ours_two_stage.py to server and run with --personalize-mode."""
import paramiko
import time
import os
import sys
from pathlib import Path

HOST = "192.168.1.85"
PORT = 22
USER = "zmx"
PASS = "Zmx20041103"
LOCAL = r"D:\ZMX\Mycode\EXPORT_PUPIL_ALL\train_ours_two_stage.py"
REMOTE = "/home/zmx/AR_Base_Data/train_ours_two_stage.py"

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "none"
    session_name = f"personalize-{mode}"
    log_file = f"train_{mode}.log"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, PORT, USER, PASS, timeout=15)
    print(f"Connected to {HOST}")

    # Upload file via SFTP
    sftp = ssh.open_sftp()
    sftp.put(LOCAL, REMOTE)
    sftp.close()
    print(f"Uploaded: {LOCAL} -> {REMOTE}")

    # Kill existing tmux session if any
    _, stdout, _ = ssh.exec_command(f"tmux kill-session -t {session_name} 2>/dev/null; echo done")
    stdout.read()
    time.sleep(0.5)

    # Create new tmux session and start training
    cmd = (
        f"tmux new-session -d -s {session_name} && "
        f"tmux send-keys -t {session_name} "
        f"'cd /home/zmx/AR_Base_Data && "
        f"python -u train_ours_two_stage.py --personalize-mode {mode} --img_size 240 --stride 4 "
        f"--epochs_stage1 6 --epochs_stage2 9 2>&1 | tee {log_file}' Enter"
    )
    _, stdout, stderr = ssh.exec_command(cmd)
    stdout.channel.recv_exit_status()
    err = stderr.read().decode()
    if err:
        print(f"stderr: {err}")
    time.sleep(1)

    # Verify session is running
    _, stdout, _ = ssh.exec_command("tmux ls 2>&1")
    out = stdout.read().decode()
    print(f"TMUX sessions:\n{out}")

    ssh.close()
    print(f"Training started in tmux session '{session_name}' (mode={mode})")
    print(f"  Check: ssh zmx@{HOST} 'tmux attach -t {session_name}'")
    print(f"  Log:   ssh zmx@{HOST} 'tail -f /home/zmx/AR_Base_Data/{log_file}'")

if __name__ == "__main__":
    main()
