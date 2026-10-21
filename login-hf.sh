#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HF_HOME="$PWD/.cache/huggingface"
exec .venvs/qwen/bin/hf auth login
