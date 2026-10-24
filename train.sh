#!/usr/bin/env bash
set -e
python3 train/build_model.py
python3 train/build_bin.py
