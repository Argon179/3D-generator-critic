#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Базовое окружение не установлено. См. README.md.' >&2
  exit 1
fi
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
export U2NET_HOME="$PWD/.cache/rembg"
export PYTHONUNBUFFERED=1
exec .venv/bin/python -m uvicorn gai.server:app --host 127.0.0.1 --port "${GAI_PORT:-7860}" --no-access-log
