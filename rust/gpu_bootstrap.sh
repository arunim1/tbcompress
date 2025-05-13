#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential clang pkg-config python3-dev libssl-dev 


curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

echo "▶ installing Rust (stable) & maturin"
if ! command -v rustup >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  . "$HOME/.cargo/env"
fi

git switch rusty

echo "▶ creating uv venv"
uv venv --python=3.12
source .venv/bin/activate

uv pip install maturin torch numpy uvicorn


mkdir -p Syzygy345_WDL
cd Syzygy345_WDL
wget -e robots=off -r -np -l1 -nH --cut-dirs=3 -A '.rtbw' https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/
cd ..


echo "▶ building universal2 wheel"
cd rust_src
maturin develop --release 
cd ..


echo "✅ ready – run: python python_demo/demo.py"
