#!/usr/bin/env bash
# One-command environment for the Holarchic Digital Twin for TAVR.
#
# Target: Ubuntu 24.04 (native, or WSL2 on Windows). Run as root (sudo) once:
#     sudo bash env/setup_ubuntu.sh            # everything
#     sudo bash env/setup_ubuntu.sh --no-segmentation   # skip TotalSegmentator (saves ~2 GB)
#
# Installs, at fixed versions, everything the twin's holons call:
#   * FEBio 4 (structural deployment), built against Intel MKL 2024.2 for the Pardiso solver
#   * svMultiPhysics (paravalvular CFD), built with MPI
#   * Python venv with tavr_decide (+ TotalSegmentator unless --no-segmentation)
# and checks each one with a real run. Every trap found while building this by hand is
# encoded here, with the reason, so nobody has to rediscover it:
#   - MKL 2026 removed mkl_dcsrtrsv, which FEBio's preconditioners still call: use 2024.2.
#   - FEBio's shared libraries use OpenMP but the executable is linked without it:
#     add -fopenmp to both linker flag sets, or the link fails on GOMP_* symbols.
#   - Ubuntu's libvtk9-dev CMake config demands Qt5 even when nothing uses it.
#   - FEBio's automatic MKL detection does not find oneAPI apt installs: pass paths.
#
# Paths are the ones tavr_decide's runners expect (/opt/src/...); keep them.
set -euo pipefail

FEBIO_COMMIT=74f111f456577a80aec65870d9c03982300324a8      # 2026-10-05, FEBio 4.13
SVMP_COMMIT=d18b32339c95b839dc797e572ba46eac948c1af7       # 2026-09-29
MKL_VER=2024.2
REPO_URL=${REPO_URL:-https://github.com/expelius/Holarchic-Digital-Twin.git}
JOBS=${JOBS:-$(nproc)}
WITH_SEG=1
for a in "$@"; do [ "$a" = "--no-segmentation" ] && WITH_SEG=0; done

LOG=/opt/src/setup_logs; mkdir -p "$LOG"
step() { echo; echo "==> $*"; }
run() { local name=$1; shift; if ! "$@" > "$LOG/$name.log" 2>&1; then echo "FAILED: $name (see $LOG/$name.log)"; tail -20 "$LOG/$name.log"; exit 1; fi; }
export DEBIAN_FRONTEND=noninteractive

step "1/7 system packages"
run apt_update apt-get update
run apt_base apt-get install -y build-essential cmake git wget gpg ca-certificates python3 python3-pip python3-venv \
    libomp-dev libopenmpi-dev openmpi-bin liblapack-dev libblas-dev libvtk9-dev qtbase5-dev libqt5opengl5-dev

step "2/7 Intel MKL $MKL_VER (Pardiso for FEBio)"
if [ ! -d /opt/intel/oneapi/mkl/$MKL_VER ]; then
  wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list
  run apt_update_intel apt-get update
  run apt_mkl apt-get install -y intel-oneapi-mkl-devel-$MKL_VER
fi
MKLROOT=/opt/intel/oneapi/mkl/$MKL_VER
IOMP=$(ls /opt/intel/oneapi/compiler/$MKL_VER/lib/libiomp5.so 2>/dev/null || ls /opt/intel/oneapi/compiler/*/lib/libiomp5.so | head -1)
[ -f "$MKLROOT/include/mkl.h" ] && [ -f "$IOMP" ] || { echo "MKL $MKL_VER or libiomp5 not found"; exit 1; }
MKL_LIBS="$MKLROOT/lib:$(dirname "$IOMP")"

step "3/7 FEBio @ ${FEBIO_COMMIT:0:10}"
mkdir -p /opt/src && cd /opt/src
[ -d FEBio ] || run febio_clone git clone https://github.com/febiosoftware/FEBio.git
cd FEBio && run febio_checkout git checkout -q $FEBIO_COMMIT
mkdir -p build && cd build
run febio_cmake cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=ON -DMKL_INC=$MKLROOT/include -DMKL_LIB_DIR=$MKLROOT/lib \
    -DMKL_OMP_LIB=$IOMP -DUSE_HYPRE=OFF -DUSE_MMG=OFF -DUSE_LEVMAR=OFF -DUSE_ZLIB=OFF -DUSE_SSH=OFF -DUSE_TETGEN=OFF \
    -DUSE_ITK=OFF -DCMAKE_EXE_LINKER_FLAGS=-fopenmp -DCMAKE_SHARED_LINKER_FLAGS=-fopenmp
run febio_make make -j"$JOBS"
cat > /usr/local/bin/febio4 <<EOF
#!/usr/bin/env bash
export LD_LIBRARY_PATH=$MKL_LIBS:\${LD_LIBRARY_PATH:-}
exec /opt/src/FEBio/build/bin/febio4 "\$@"
EOF
chmod +x /usr/local/bin/febio4

step "4/7 svMultiPhysics @ ${SVMP_COMMIT:0:10}"
cd /opt/src
[ -d svMultiPhysics ] || run svmp_clone git clone https://github.com/SimVascular/svMultiPhysics.git
cd svMultiPhysics && run svmp_checkout git checkout -q $SVMP_COMMIT
mkdir -p build && cd build
run svmp_cmake cmake .. -DCMAKE_BUILD_TYPE=Release
run svmp_make make -j"$JOBS"
ln -sf /opt/src/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics /usr/local/bin/svmultiphysics

step "5/7 Python environment"
python3 -m venv /opt/tavr/venv
run pip_upgrade /opt/tavr/venv/bin/pip install --upgrade pip
[ -d /opt/tavr/src ] || run repo_clone git clone "$REPO_URL" /opt/tavr/src
run pip_pkg /opt/tavr/venv/bin/pip install -e "/opt/tavr/src[dev,dicom]"
if [ "$WITH_SEG" = 1 ]; then
  run pip_torch /opt/tavr/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  run pip_seg /opt/tavr/venv/bin/pip install totalsegmentator
fi
ln -sf /opt/tavr/venv/bin/tavr-decide /usr/local/bin/tavr-decide

step "6/7 checks with real runs"
WORK=$(mktemp -d)
cat > $WORK/cube.feb <<'EOF'
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0"><Module type="solid"/>
<Control><analysis>STATIC</analysis><time_steps>10</time_steps><step_size>0.1</step_size>
<solver type="solid"><qn_method type="BFGS"/></solver></Control>
<Material><material id="1" name="m" type="neo-Hookean"><E>1</E><v>0.3</v></material></Material>
<Mesh><Nodes name="all"><node id="1">0,0,0</node><node id="2">1,0,0</node><node id="3">1,1,0</node><node id="4">0,1,0</node>
<node id="5">0,0,1</node><node id="6">1,0,1</node><node id="7">1,1,1</node><node id="8">0,1,1</node></Nodes>
<Elements type="hex8" name="P"><elem id="1">1,2,3,4,5,6,7,8</elem></Elements>
<NodeSet name="bot">1,2,3,4</NodeSet><NodeSet name="top">5,6,7,8</NodeSet></Mesh>
<MeshDomains><SolidDomain name="P" mat="m"/></MeshDomains>
<Boundary><bc name="f" node_set="bot" type="zero displacement"><x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof></bc>
<bc name="p" node_set="top" type="prescribed displacement"><dof>z</dof><value lc="1">0.2</value></bc></Boundary>
<LoadData><load_controller id="1" type="loadcurve"><points><pt>0,0</pt><pt>1,1</pt></points></load_controller></LoadData>
</febio_spec>
EOF
(cd $WORK && febio4 -i cube.feb -silent > /dev/null 2>&1 || true)
grep -q "N O R M A L" $WORK/cube.log && grep -q "pardiso" $WORK/cube.log && echo "  FEBio: normal termination, Pardiso solver" \
  || { echo "  FEBio check FAILED"; tail -20 $WORK/cube.log; exit 1; }
cd /opt/tavr/src
run pytest /opt/tavr/venv/bin/python -m pytest -q
echo "  tavr_decide tests: $(tail -1 $LOG/pytest.log)"
run cfd_verify /opt/tavr/venv/bin/python runs/cfd_verify.py $WORK/cfd 3 60
echo "  CFD Poiseuille check: $(grep 'CFD/analytic' $LOG/cfd_verify.log | sed 's/.*CFD\/analytic/CFD\/analytic/')"

step "7/7 done"
cat <<EOF
Installed:
  febio4          -> FEBio 4 with Pardiso (MKL $MKL_VER)
  svmultiphysics  -> svMultiPhysics (MPI)
  tavr-decide     -> command line of the twin  (python: /opt/tavr/venv/bin/python)
Logs: $LOG
EOF
