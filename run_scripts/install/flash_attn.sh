#!/usr/bin/env bash
# flash-attn for the pixi ``quantize`` env. The last release predates torch 2.14
# and sm_110, so a main commit is built with its ``-std=c++17`` flags raised to
# c++20 (torch 2.14 headers need std::strong_ordering). Wheels are cached per
# torch / Python / arch, so a reinstall skips the compile.
set -euo pipefail

FA_COMMIT="${FA_COMMIT:-ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820}"  # main 2026-08-28, 2.8.4

case "$(uname -m)" in
  aarch64) default_archs="110;120" ;;      # Thor, GB10 (sm_121 runs the sm_120 build)
  *)       default_archs="90;100;120" ;;   # H100, B200, RTX 5090
esac
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-$default_archs}"

torch_version="$(python -c 'import torch; print(torch.__version__)')"
python_tag="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
wheel_dir="${XDG_CACHE_HOME:-$HOME/.cache}/rldx/flash-attn/${FA_COMMIT:0:9}-torch${torch_version/+/_}-${python_tag}-sm${FLASH_ATTN_CUDA_ARCHS//;/_}"

if ! ls "$wheel_dir"/flash_attn-*.whl >/dev/null 2>&1; then
  src="$(mktemp -d)"
  trap 'rm -rf "$src"' EXIT
  git clone --quiet https://github.com/Dao-AILab/flash-attention.git "$src"
  git -C "$src" checkout --quiet "$FA_COMMIT"
  git -C "$src" submodule update --init --quiet csrc/cutlass
  sed -i 's/-std=c++17/-std=c++20/g' "$src/setup.py"
  mkdir -p "$wheel_dir"
  FLASH_ATTENTION_FORCE_BUILD=TRUE pip wheel --no-deps --no-build-isolation -w "$wheel_dir" "$src"
fi

pip install --no-deps --force-reinstall "$wheel_dir"/flash_attn-*.whl
