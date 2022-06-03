#!/usr/bin/env sh

set -e

# Ensure current path is project root
cd "$(dirname "$0")/../"


poetry run sphinx-apidoc --force --separate --no-toc -o docs/source quaterion
poetry build -f wheel
pip install dist/$(ls -1 dist | grep .whl)
poetry run sphinx-build docs/source docs/html
