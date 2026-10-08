#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || python -m pip install --user "uv>=0.10,<1"
command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 .venv
# Pin CUDA 12.6 instead of accepting an uncontrolled future CUDA/PyTorch release.
uv pip install --python .venv/bin/python --torch-backend=cu126 \
  -r requirements.txt
.venv/bin/python - <<'PY'
import torch, gradio
from voxcpm import VoxCPM
print("PyTorch:", torch.__version__)
print("Gradio:", gradio.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
echo "Ejecuta: .venv/bin/python -u studio.py --share"
