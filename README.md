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
- High prediction accuracy
- Easy-to-use command line and Python API
- GPU acceleration support

<div align="center">
  <img src="Fig/Fig1.jpg" alt="DeepMp Framework" width="90%">
  <p><em>DeepMp framework for microprotein identification. (A) Model architecture combining CNN, Bi-GRU, and attention mechanisms. (B-C) Performance comparison showing DeepMp outperforms established language models (ALBERT V2, BERT, RoBERTa, Transformer) with superior accuracy.</em></p>
</div>

## Installation

### Method 1: Using environment.yml File

```bash
# 1. Create environment from yml file
conda env create -f environment.yml

# 2. Activate the environment
conda activate DeepMp

# 3. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"

```

### Method 2: Direct Conda Installation
```bash

# 1. Create new environment with Python 3.9
conda create -n DeepMp python=3.9 -y

# 2. Activate the environment
conda activate DeepMp

# 3. Install all dependencies (single command)
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 numpy=1.24.3 pandas=2.1.3 scipy=1.11.3 scikit-learn=1.3.1 matplotlib=3.8.0 seaborn=0.13.0 biopython=1.81 tqdm=4.66.1 joblib=1.3.2 -c pytorch -c conda-forge -y

# 4. Verify installation
python -c "import torch, numpy; print(f'PyTorch: {torch.__version__}, NumPy: {numpy.__version__}')"
```
## Usage

### Train Model
- Positive Dataset: A total of 13,767 validated plant peptide sequences. The sources are: 6,828 and 6,816 entries from Arabidopsis and rice, respectively. An additional 123 peptides expanded from public databases.
- Negative Dataset: Constructed based on the following criteria: Selected Open Reading Frames (ORFs) lacking support from omics data. Ensured these negative sequences share no homology with the positive dataset mentioned above to prevent bias.
```bash
python Script/4CNN_BiLGRU_Attention_train.py -input ./Data/Data.csv
```
### Predict Microproteins
- Model Applicability Note:
This model is primarily designed for predicting microproteins in plant species that are closely related to Arabidopsis and rice.

- Important:
For species that are more distantly related (e.g., animals, fungi, or distant plant lineages), it is recommended to first collect species-specific verified microprotein data to re-train or fine-tune the model before making predictions. Direct application may lead to reduced accuracy due to sequence feature divergence.
```bash
python Script/4CNN_BiGRU_Attention_predict.py -input ./Data/Test.fasta -output ./results/Test_predict_result.txt
```
## Citation

Jia, Anqiang, et al. Deep learning reveals a microprotein atlas in maize and uncovers novel regulators of seed amino acid metabolism. bioRxiv (2025): 2025-11. https://doi.org/10.1101/2025.11.14.688563.
