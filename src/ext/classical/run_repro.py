"""Run the Barath repo's own TUM-RS real-data example (examples/run_real_data.py -> tester_rs_superansac.py) with
the romatch shim (roma_compat.py) applied, without editing any third-party file.

The arguments are the ones examples/run_real_data.py builds by default, plus --workers 4 (CPU budget; the tester
defaults to os.cpu_count()).
usage: python run_repro.py --sequences 1 --output <csv> [any extra tester flags]
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roma_compat  # noqa: E402

roma_compat.patch()

REPO = os.path.expanduser("~/rs3d/third_party/rolling_shutter_made_practical")
TESTER = f"{REPO}/external/superansac/tests/essential_matrix/tester_rs_superansac.py"
defaults = ["--dataset", "tum_rs", "--dataset_root", os.path.expanduser("~/rs3d/ext/classical/tum_rs_link"),
            "--strides", "10", "20", "--max_pairs", "100", "--features", "RoMa",
            "--methods", "ac_rs_7direct", "gs_5pt", "dai_20pt", "dai_44pt", "--scoring", "MSAC",
            "--sampler", "PROSAC", "--threshold", "1.0", "--max_iterations", "1000", "--device", "cuda",
            "--workers", "4"]
sys.argv = [TESTER] + defaults + sys.argv[1:]
os.chdir(os.path.dirname(TESTER))
sys.path.insert(0, os.path.dirname(TESTER))  # as when the tester is run as a script
runpy.run_path(TESTER, run_name="__main__")
