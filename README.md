# 🧬 DeepMp: A deep learning framework for genome-wide microprotein identification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A deep learning model for microprotein identification (5-100 amino acids)**

[Overview](#-overview) | [Installation](#-installation) | [Usage](#usage) | [Citation](#citation)

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

### Installation Dependencies
```bash
# Version-tested compatible combination for DeepMp
# PyTorch ecosystem
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Numerical computing and data processing
pip install numpy==1.24.3        # Compatible with PyTorch 2.1.0
pip install pandas==2.1.3        # Latest stable with good compatibility
pip install scipy==1.11.3        # Compatible with numpy 1.24.3

# Machine learning utilities
pip install scikit-learn==1.3.1  # Compatible with numpy 1.24.3

# Visualization
pip install matplotlib==3.8.0    # Compatible with numpy 1.24.3
pip install seaborn==0.13.0      # Requires matplotlib >= 3.6

# Bioinformatics
pip install biopython==1.81      # Compatible with numpy 1.24.3

# Utilities
pip install tqdm==4.66.1
pip install joblib==1.3.2        # For scikit-learn model persistence

```
## Installation

### Train Model
```bash
python Script/4CNN_BiLGRU_Attention_train.py
```
### Predict Microproteins
```bash
python Script/4CNN_BiLSTM_Attention_predict.py
```
## Citation

Jia, Anqiang, Yawen Yang, Min Jin, Jimin Zhan, Mi Zhang, Sixuan Xu, Zhen Li et al. Deep learning reveals a microprotein atlas in maize and uncovers novel regulators of seed amino acid metabolism. bioRxiv (2025): 2025-11.
