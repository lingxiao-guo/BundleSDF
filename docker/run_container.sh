#!/usr/bin/env bash
set -e

docker rm -f bundlesdf >/dev/null 2>&1 || true
DIR=$(pwd)/../

DISPLAY_ARGS=()
if command -v xhost >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
  xhost +local:root >/dev/null 2>&1 || true
  DISPLAY_ARGS=(-e DISPLAY="${DISPLAY}")
else
  echo "xhost not available or DISPLAY not set; skipping X11 access."
fi

DRI_ARGS=()
if [ -e /dev/dri ]; then
  DRI_ARGS=(--device=/dev/dri)
fi

docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it \
  --network=host --name bundlesdf --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined -v /home:/home -v /tmp:/tmp -v /mnt:/mnt \
  -v /usr/share/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d:ro \
  -v "$DIR:$DIR" --ipc=host \
  "${DISPLAY_ARGS[@]}" "${DRI_ARGS[@]}" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  nvcr.io/nvidian/bundlesdf:latest \
  bash -lc "pip install yacs && pip install \"trimesh<4\" && pip install warp-lang pygltflib ipycanvas ipyevents pybind11 usd-core \"jupyter_client<8\" && cd \"$DIR/mycuda\" && rm -rf build *egg* && pip install -e .; exec bash"
