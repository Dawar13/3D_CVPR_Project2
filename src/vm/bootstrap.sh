#!/bin/bash
# EXP2 on a fresh VM in one command. Before it (from the laptop): the big disk mounted, ~/rs3d a git checkout of
# this repo (`git push vm main`), and the ADT URL file copied to ~/rs3d/data/adt/. Then on the VM:
#   tmux new-session -d -s rs3d -n boot "bash ~/rs3d/src/vm/bootstrap.sh; exec bash"
# 1. system packages (background), uv, the ~/rs3d/ext link to src/ext
# 2. everything network-bound at once: main env + weights (setup_env.sh), every other env (setup_extra_envs.sh),
#    TUM RS-VIO, Replica, ADVIO, ETH3D, ADT, Waymo (after the user's login)
# 3. every driver (run_all.sh): the GPU starts with the EXP0 reproduction as soon as the main env, its weights and
#    TUM RS-VIO are in; every other driver joins when its inputs land.
# Log: logs/bootstrap.log
source ~/rs3d/src/vm/lib.sh
exec > >(tee -a $L/bootstrap.log) 2>&1
echo "$(stamp) bootstrap: $(nproc) CPUs, $(free -g | awk '/Mem/{print $2}') GB RAM, git $(git -C $R rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
df -h $R/ | tail -n 1
mkdir -p $R/{data,cache,results/exp2,third_party,bin,envs,weights,locks,.hf} $R/data/adt
[ -e $R/ext ] || ln -s $R/src/ext $R/ext

if sudo -n true 2>/dev/null; then
  (sudo -n apt-get update -qq && sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -qq pigz unzip cmake \
     libeigen3-dev libboost-all-dev libopencv-dev libopencv-contrib-dev && echo APT_DONE || echo APT_FAILED) > $L/apt.log 2>&1 &
else
  echo "APT_FAILED (no passwordless sudo)" > $L/apt.log
fi
[ -x $R/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$R/bin UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh

bash $R/src/download_tum.sh > /dev/null 2>&1 &
bash $R/src/setup_env.sh > /dev/null 2>&1 &
bash $R/src/vm/setup_extra_envs.sh > /dev/null 2>&1 &
( until [ -s $L/apt.log ]; do sleep 5; done; bash $R/src/download_more.sh replica > /dev/null 2>&1 ) &  # unpigz
bash $R/src/download_more.sh advio > /dev/null 2>&1 &
bash $R/src/download_more.sh eth3d > /dev/null 2>&1 &
bash $R/src/vm/download_adt.sh &
bash $R/src/vm/download_waymo.sh &
bash $R/src/vm/run_all.sh
echo "$(stamp) bootstrap: everything launched"
wait
echo "$(stamp) BOOTSTRAP_DONE (envs and downloads finished; drivers keep running in their windows)"
