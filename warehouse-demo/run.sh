#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  "${PYTHON:-python3.12}" -m venv .venv
  .venv/bin/python -m pip install -r requirements.lock
fi
if [ ! -f menagerie/unitree_go2/go2.xml ]; then
  if [ ! -d menagerie/.git ]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git menagerie
  fi
  git -C menagerie fetch --depth 1 origin 4d038b3feae26ec82b46a4d586379114012a8ac7
  git -C menagerie checkout 4d038b3feae26ec82b46a4d586379114012a8ac7
  git -C menagerie sparse-checkout set unitree_go2
fi
if [ ! -f runtime/items-yolo.pt ]; then
  .venv/bin/python train_items.py
fi
exec .venv/bin/python demo.py
