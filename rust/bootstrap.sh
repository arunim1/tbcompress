#!/usr/bin/env bash
set -euo pipefail
# echo "▶ creating uv venv"
# uv venv --python=3.12
source .venv/bin/activate

echo "▶ installing Rust (stable) & maturin"
if ! command -v rustup >/dev/null; then
  brew install rustup-init
  rustup-init -y --profile minimal
fi
rustup target add aarch64-apple-darwin

uv pip install maturin 'torch>=2' numpy uvicorn  # torch pulls Metal backend on Apple Silicon

echo "▶ building universal2 wheel"
cd rust_src
maturin develop --release --target universal2-apple-darwin  # ≥ 0.14 auto-merges x86_64+aarch64 
cd ..

echo "✅ ready – run: python python_demo/demo.py"
