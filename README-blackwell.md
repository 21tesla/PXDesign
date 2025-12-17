# PXDesign for NVIDIA Blackwell (RTX 6000 / B200)

This repository contains an automated installer to run **PXDesign** on NVIDIA Blackwell architecture (sm_120), which requires PyTorch Nightly, CUDA 12.8+, and JAX updates that break the standard installation.

## Prerequisites
* Linux
* CUDA 12.8+ Driver
* Conda

## Installation
1. Clone this repo.
2. Run the installer:
   ```bash
   bash install_blackwell.sh
