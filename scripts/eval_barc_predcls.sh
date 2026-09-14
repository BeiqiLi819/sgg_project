#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/star_barc_predcls.py}" \
  bash "$(dirname "$0")/eval_once.sh"
