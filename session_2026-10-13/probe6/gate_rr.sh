#!/bin/bash
# gate_rr.sh (2026-10-13, Claude): CPU gate for the RadImageNet R50 probe (PROBE6_PREREG.md); called once by the queue runner.
cd ~/oai_build || exit 1
CUDA_VISIBLE_DEVICES= nice -n 19 ~/miniconda3/envs/rsna/bin/python gate_rr.py
