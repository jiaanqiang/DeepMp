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
pip install deepmp
