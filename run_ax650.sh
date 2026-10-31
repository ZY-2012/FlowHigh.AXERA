#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ "$#" -ge 2 ] && [ "${1#-}" = "$1" ]; then
  IN="$1"; OUT="$2"; shift 2
  exec python3 python/demo_board.py --models models --steps 1 --std 0.0001 --input "$IN" --output "$OUT" "$@"
fi
exec python3 python/demo_board.py --models models --steps 1 --std 0.0001 "$@"
