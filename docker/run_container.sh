#!/usr/bin/env bash
set -e

docker rm -f bundlesdf >/dev/null 2>&1 || true
DIR=$(pwd)/../

DISPLAY_ARGS=()
if command -v xhost >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
  xhost +local:root >/dev/null 2>&1 || true
  DISPLAY_ARGS=(-e DISPLAY="${DISPLAY}" --device=/dev/dri)
else
  echo "xhost not available or DISPLAY not set; skipping X11 access."
fi

docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it \
  --network=host --name bundlesdf --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined -v /home:/home -v /tmp:/tmp -v /mnt:/mnt \
  -v "$DIR:$DIR" --ipc=host \
  "${DISPLAY_ARGS[@]}" \
  -e GIT_INDEX_FILE nvcr.io/nvidian/bundlesdf:latest \
  bash -lc "pip install yacs && pip install \"trimesh<4\" && pip install warp-lang pygltflib ipycanvas ipyevents pybind11 usd-core \"jupyter_client<8\" && cd /home/ubuntu/projects/real2sim/BundleSDF/mycuda && rm -rf build *egg* && pip install -e .; exec bash"
