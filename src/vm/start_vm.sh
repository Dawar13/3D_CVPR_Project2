#!/bin/bash
# Run on the LAPTOP (Git Bash) the moment a new VM is up:  bash src/vm/start_vm.sh <ip> [user]
# 1. storage: ~/rs3d goes on the big disk (/data if mounted, else the home disk if it has >= 1.2 TB free;
#    otherwise stop and print lsblk: formatting a disk is done by hand, after checking it is blank)
# 2. ~/rs3d becomes a git checkout of this repo (push to the `vm` remote, receive.denyCurrentBranch=updateInstead)
# 3. the user's ADT URL file (signed links, never committed) goes to ~/rs3d/data/adt/
# 4. bootstrap.sh starts in tmux session rs3d, window boot
set -euo pipefail
IP=$1; U=${2:-gpunet}
H=$U@$IP
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ssh -o StrictHostKeyChecking=accept-new $H 'set -e
  if [ -e ~/rs3d ]; then echo "~/rs3d exists"
  elif mountpoint -q /data; then sudo -n mkdir -p /data/rs3d && sudo -n chown $(id -u):$(id -g) /data/rs3d && ln -s /data/rs3d ~/rs3d
  elif [ $(df --output=avail -B1G ~ | tail -n 1) -ge 1200 ]; then mkdir ~/rs3d
  else echo "NO_BIG_DISK: mount the data disk first"; lsblk -f; df -h; exit 3; fi
  cd ~/rs3d && [ -d .git ] || git init -q -b main . && git config receive.denyCurrentBranch updateInstead
  df -h ~/rs3d/ | tail -n 1; nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
cd "$REPO"
git remote get-url vm > /dev/null 2>&1 && git remote set-url vm $H:rs3d || git remote add vm $H:rs3d
git push -q vm main
ssh $H 'mkdir -p ~/rs3d/data/adt'
scp -q ADT_download_urls.json $H:rs3d/data/adt/ADT_download_urls.json
ssh $H 'chmod 600 ~/rs3d/data/adt/ADT_download_urls.json
  tmux has-session -t rs3d 2>/dev/null || tmux new-session -d -s rs3d -n boot "bash ~/rs3d/src/vm/bootstrap.sh; exec bash"
  sleep 5; tail -n 5 ~/rs3d/logs/bootstrap.log'
echo "started on $H at $(date -u +%FT%TZ)"
