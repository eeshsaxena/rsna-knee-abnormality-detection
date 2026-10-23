#!/bin/bash
# gate_sw.sh (2026-10-12, Claude): CPU gate for the Swin-T probe (PROBE5_PREREG.md); called once by the queue runner.
cd ~/oai_build || exit 1
CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/rsna/bin/python gate_sw.py
