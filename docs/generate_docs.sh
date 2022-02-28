#!/usr/bin/env sh

set -e

# Ensure current path is project root
cd "$(dirname "$0")/../"


poetry run sphinx-apidoc -f -e -o docs/source quaterion
poetry run sphinx-build docs/source docs/html
