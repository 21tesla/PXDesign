# PXDesign for NVIDIA Blackwell

This repository contains an automated installer to run **PXDesign** on NVIDIA Blackwell architecture (sm_120), which requires PyTorch Nightly, CUDA 12.8+, and JAX updates that break the standard installation.

## Prerequisites
* CUDA 12.8+ Driver
* Conda
* Tested on Ubuntu 24.04 + Blackwell RTX 6000 (driver 580.105.08 / cuda 13.0)

## Installation
1. Clone this repo.
2. Run the installer:
   ```bash
   bash install_blackwell.sh

## Environment
   ```bash

cd $HOME/PXDesign

conda activate pxdesign

pxdesign pipeline \
    --preset extended \
    -i ./examples/PDL1_quick_start.yaml \
    -o ./output_folder \
    --N_sample 10 \
    --dtype bf16 \
    --use_fast_ln False \
    --use_deepspeed_evo_attention False
