#!/bin/bash
# Start every EXP2 driver in its own window of tmux session rs3d. Each driver waits for exactly the inputs it
# needs (envs, downloads, earlier stages), so they can all be started at once at boot.
# GPU sharing: main drivers share RS3D_GPU_SLOTS (3) slots; extra models 2, unrollers 1, Barath 1, LoRA evaluation 1,
# LoRA training outside the slots (gpu_slot.sh reserves its memory while it waits). Every slot needs >= 14 GB free.
source ~/rs3d/src/vm/lib.sh
export RS3D_GPU_SLOTS=${RS3D_GPU_SLOTS:-3}
win() {  # win <name> <script>: driver output also goes to logs/driver_<name>.out
  tmux new-window -d -t rs3d -n $1 "bash $R/src/$2 2>&1 | tee -a $L/driver_$1.out; echo \"EXIT \$(date -u +%FT%TZ)\" >> $L/driver_$1.out; exec bash"
}
win stage1 run_exp2_stage1.sh
win tum2 run_exp2_tum2.sh
win gen run_synth_gen.sh
win syncore run_synth_core.sh
win realgap run_real_gap.sh
win adt run_adt.sh
win waymo run_waymo.sh
win noharm run_noharm.sh
win stress run_stress.sh
win models vm/run_models.sh
win u2d vm/run_u2d.sh
win classical vm/run_classical.sh
win lora vm/run_lora.sh
win final vm/run_final.sh
echo "$(stamp) run_all: $(tmux list-windows -t rs3d | wc -l) windows"
