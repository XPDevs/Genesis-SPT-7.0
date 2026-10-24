#!/usr/bin/env bash
set -e
python3 train/generate_training_data.py
python3 train/build_model.py
python3 train/build_bin.py
python3 train/validate.py
echo "DONE"
