#!/bin/bash
# setup_env.sh: STUDIO side (free CPU Studio, zero credits). Builds ~/rsna_d2/venv = Python 3.11 + the BOX pins
# (torch 2.5.1+cu121, torchvision 0.20.1+cu121 from the PyTorch cu121 index; numpy 1.26.4, pandas 3.0.5,
# scikit-learn 1.9.0, scipy 1.17.1, timm 1.0.29, huggingface_hub 0.36.2, safetensors 0.8.0, PyYAML 6.0.3,
# pillow 12.3.0 from PyPI; opencv-python-headless 4.14.0.94 with --no-deps, exactly as on the box, Vast and Modal).
# The venv lives in the Studio filesystem, so the GPU job (Studio snapshot) gets it unchanged.
set -eu
# Lightning Studios set UV_LINK_MODE=symlink (packages live only in the uv cache under /system/conda): force real
# copies so the venv is self-contained, and NEVER run `uv cache clean` (on 10-09 that emptied the first venv).
export UV_LINK_MODE=copy
V=~/rsna_d2/venv; P=$V/bin/python
echo "[env $(date '+%F %T')] uv $(uv --version)"
[ -x $P ] || uv venv --python 3.11 $V
$P -c 'import sys; print("python", sys.version)'
uv pip install --python $P --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1+cu121" "torchvision==0.20.1+cu121"
cat > ~/rsna_d2/constraints.txt <<EOF
torch==2.5.1+cu121
torchvision==0.20.1+cu121
EOF
uv pip install --python $P --index-url https://pypi.org/simple -c ~/rsna_d2/constraints.txt \
  numpy==1.26.4 pandas==3.0.5 scikit-learn==1.9.0 scipy==1.17.1 timm==1.0.29 huggingface_hub==0.36.2 \
  safetensors==0.8.0 PyYAML==6.0.3 pillow==12.3.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match
uv pip install --python $P --index-url https://pypi.org/simple --no-deps opencv-python-headless==4.14.0.94
uv pip freeze --python $P | grep -i -E '^(opencv-python-headless|numpy|pandas|scikit-learn|scipy|timm|huggingface[-_]hub|safetensors|pyyaml|pillow|torch|torchvision)==' | sort -f > ~/rsna_d2/env_pins.txt
cat ~/rsna_d2/env_pins.txt
$P -c 'import torch, torchvision, cv2, numpy, pandas, sklearn, scipy, timm; print("IMPORT_OK torch", torch.__version__, "cuda build", torch.version.cuda, "cudnn", torch.backends.cudnn.version(), "cv2", cv2.__version__)'
echo "[env $(date '+%F %T')] ENV_DONE"
