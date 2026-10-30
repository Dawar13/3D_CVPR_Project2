source ~/rs3d/.venv/bin/activate
export HF_HOME=~/rs3d/.hf HF_HUB_OFFLINE=1 PYTHONPATH=~/rs3d/ext/lora RS3D_EXTRA_RUNNERS=models_lora LORA_CKPT=base
cd ~/rs3d/results/exp2/training/eval/base_check
for m in pi3 vggt; do
env RS3D_LOCK_DIR=$HOME/rs3d/locks_lora RS3D_GPU_SLOTS=1 bash ~/rs3d/src/gpu_slot.sh python ~/rs3d/src/infer2.py --model ${m}_lora --clips clips_synth.csv --out . --name synth > log_${m}_synth.txt 2>&1
env RS3D_LOCK_DIR=$HOME/rs3d/locks_lora RS3D_GPU_SLOTS=1 bash ~/rs3d/src/gpu_slot.sh python ~/rs3d/src/infer2.py --model ${m}_lora --clips clips_tum.csv --out . --name tum > log_${m}_tum.txt 2>&1
done
echo BASECHECK_DONE > done.txt
