#!/usr/bin/env sh

set -xe

# Ensure current path is project root
cd "$(dirname "$0")/../"

pip install tk
# install CPU torch, cause it is smaller
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

pip install poetry
poetry build -f wheel
pip install dist/$(ls -1 dist | grep .whl)
pip install pytorch-metric-learning==1.3.2

pip install sphinx>=5.0.1
pip install "git+https://github.com/qdrant/qdrant_sphinx_theme.git@master#egg=qdrant-sphinx-theme"

sphinx-apidoc --force --separate --no-toc -o docs/source quaterion
sphinx-build docs/source docs/html
