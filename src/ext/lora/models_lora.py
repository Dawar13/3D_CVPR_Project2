"""infer2 extra runners for the EXP2 §8.6 LoRA controls: pi3_lora, vggt_lora.

Each wraps the unmodified infer2 runner (Pi3Runner / VGGTRunner: same checkpoint, preprocessing, bf16 placement,
outputs) and injects the LoRA weights of the checkpoint named by env var LORA_CKPT into its model. LORA_CKPT=base
runs the untouched base model through the same wrapper.

usage: PYTHONPATH=~/rs3d/ext/lora RS3D_EXTRA_RUNNERS=models_lora LORA_CKPT=<.../best.pt> \
         python ~/rs3d/src/infer2.py --model pi3_lora --clips <csv> --out <dir> --name <name> [--dense]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class _LoraRunner:
    base = None  # "pi3" | "vggt"

    def __init__(self):
        import common_lora as C
        ck = os.environ.get("LORA_CKPT", "")
        if not ck:
            raise RuntimeError("set LORA_CKPT to a LoRA checkpoint (.pt) or 'base'")
        self._r = C.make_runner(self.base)  # infer2's own runner class, model on cuda in eval mode
        self.model = self._r.model
        if ck == "base":
            self.lora_meta = None
            self.ckpt = f"{self._r.ckpt} (no LoRA)"
        else:
            self.lora_meta = C.load_lora(self.model, ck)
            assert self.lora_meta["model"] == self.base, (self.lora_meta["model"], self.base)
            self.model.eval()
            self.ckpt = f"{self._r.ckpt} + LoRA {os.path.realpath(ck)} (step {self.lora_meta.get('step')})"
        print(f"[{self.base}_lora] {self.ckpt}", flush=True)

    def load(self, paths):
        return self._r.load(paths)

    def channel0(self, x):
        return self._r.channel0(x)

    def run(self, x, dense):
        return self._r.run(x, dense)


class Pi3LoraRunner(_LoraRunner):
    base = "pi3"


class VGGTLoraRunner(_LoraRunner):
    base = "vggt"


EXTRA_RUNNERS = {"pi3_lora": Pi3LoraRunner, "vggt_lora": VGGTLoraRunner}
