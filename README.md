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

Jia, Anqiang, et al. Deep learning reveals a microprotein atlas in maize and uncovers novel regulators of seed amino acid metabolism. bioRxiv (2025): 2025-11.
