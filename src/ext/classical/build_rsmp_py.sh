#!/usr/bin/env bash
# Python module (pysuperansac) for Barath RS-7AC; same as `pip install ./external/superansac` (setup.py runs
# cmake + make), with the overlay of RS files applied first as the repo intends. Recovered from the helper's
# transcript after the first VM was terminated; the C++ synthetic benchmark build is not needed and is skipped.
set -euxo pipefail
R=~/rs3d/third_party/rolling_shutter_made_practical
PY=~/rs3d/envs/rsmp/bin/python
J=$(( $(nproc) / 2 ))
cd $R
bash rs_overlay/apply_overlay.sh
OUTLIB=$R/build_py/lib
mkdir -p $OUTLIB
cmake -S external/superansac -B build_py/tmp -DCMAKE_BUILD_TYPE=Release -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$OUTLIB \
  -DPYTHON_EXECUTABLE=$PY -DPython_EXECUTABLE=$PY -Dpybind11_DIR=$($PY -m pybind11 --cmakedir)
cmake --build build_py/tmp --config Release -- -j$J
SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo $OUTLIB > $SP/pysuperansac_local.pth
$PY -c "import pysuperansac; print('RS7AC:', hasattr(pysuperansac, 'estimateRSEssentialMatrix7AC'))"
echo PY_BUILD_OK
