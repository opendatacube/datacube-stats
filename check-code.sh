#!/usr/bin/env bash
# Mimic Travis CI checks

set -eu
set -x

pep8 datacube_stats scripts tests --max-line-length=120

pylint -j 2 --reports no datacube_stats

pytest -vv -r sx --cov datacube_stats --durations=5 datacube_stats tests $@
