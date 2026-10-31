#!/bin/bash
set -e
cd "$(dirname "$0")/.."
IMAGE="${PULSAR2_IMAGE:-pulsar2:8.0}"
HASP_ARGS=(-v /var/hasplm:/var/hasplm)
if [ -n "${HASP_HOME:-}" ]; then
  HASP_ARGS+=(-v "${HASP_HOME}:/root/.hasplm")
fi
NAMES="${@:-flowhigh_nct bigvgan_T100}"
for name in $NAMES; do
  echo "=== compile ${name} ==="
  docker run --rm --network host \
    -v "$PWD:/workspace" \
    "${HASP_ARGS[@]}" \
    -e HASP_HOME=/root/.hasplm \
    "$IMAGE" -lc "set -e; mkdir -p /root/.hasplm/installed/32434 && cp -f /root/*.v2c /root/.hasplm/installed/32434/ 2>/dev/null || true; PATH=/usr/local/bin/.venv/bin:/opt/pulsar2:\$PATH; pulsar2 build --config /workspace/model_convert/configs/${name}.json" 2>&1 | tee "build/compile_${name}.log"
done
find build/compile -name '*.axmodel' -ls
