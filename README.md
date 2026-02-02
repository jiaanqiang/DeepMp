# 🧬 DeepMp: A deep learning framework for genome-wide microprotein identification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A deep learning model for microprotein identification (5-100 amino acids)**

[Quick Start](#-quick-start) | [Installation](#-installation) | [Usage](#usage) | [Citation](#citation)

</div>

## Overview

DeepMp is a deep learning model that identifies microproteins (5-100 amino acids) from protein sequences. The model combines CNN, Bi-GRU, and Attention mechanisms for accurate prediction.

**Key Features:**
- Hybrid 4-layer CNN + 6-layer Bi-GRU + 16-head Attention architecture
- Trained on 13,497 experimentally validated microproteins
- High prediction accuracy (MCC: 0.88, Accuracy: 0.94)
- Easy-to-use command line and Python API
- GPU acceleration support

## Installation

### Quick Install
```bash
# 安装 PyTorch（根据您的CUDA版本选择）
# CPU版本
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0

# CUDA 11.8版本
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 安装其他必需包
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install biopython==1.81
pip install tqdm==4.65.0
pip install scipy==1.11.1
