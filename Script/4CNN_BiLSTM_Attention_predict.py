import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import matthews_corrcoef, confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 氨基酸到索引的映射（1-20）
amino_to_index = {amino: i+1 for i, amino in enumerate('ARNDCQEGHILKMFPSTWYV')}

# 创建自定义数据集类
class ProteinDataset(Dataset):
    def __init__(self, data, amino_to_index, max_length=100):
        self.data = data
        self.amino_to_index = amino_to_index
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # 将氨基酸序列转换为索引序列，并填充到指定的最大长度
        sequence_indices = [self.amino_to_index.get(amino, 0) for amino in sequence]
        sequence_indices += [0] * (self.max_length - len(sequence_indices))
        sequence_indices = sequence_indices[:self.max_length]

        return {
            'sequence_indices': torch.tensor(sequence_indices),
            'label': torch.tensor(label)
        }


# 定义模型
class ProteinPredictor(nn.Module):
    def __init__(self, input_size, num_classes, max_length=100):
        super(ProteinPredictor, self).__init__()

        embedding_dim = 128
        hidden_size = 128
        
        # Define the layers
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(1024, 2048, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        # Adjust the input size for GRU to match the output size of convolutions
        self.bigru = nn.GRU(2048, hidden_size, num_layers=6, batch_first=True, bidirectional=True, dropout=0.2)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=16, dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.relu = nn.ReLU()
        
        self._initialize_weights()  # Weight initialization

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.Embedding)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)  # Reshape for convolutional layer
        
        # Apply CNN layers
        x = self.conv1(embedded)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)  # 新增的卷积层
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)  # 新增的卷积层
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = x.permute(0, 2, 1)  # Reshape for GRU layer
        gru_output, _ = self.bigru(x)
        
        gru_output = gru_output.permute(1, 0, 2)  # Adjust dimensions for multihead attention
        transformer_output, _ = self.self_attention(gru_output, gru_output, gru_output)
        transformer_output = transformer_output.permute(1, 0, 2)  # Reshape back
        
        self_attention_output = self.dropout(transformer_output)
        output = self.fc(self_attention_output[:, -1, :])
        output = self.relu(output)
        
        return output

# Load the pre-trained model
model = ProteinPredictor(input_size=len(amino_to_index) + 1, num_classes=2)
model.to(torch.device('cuda'))
model.load_state_dict(torch.load('best_CNN_gpu.pth'))
model.eval()

# 加载 FASTA 文件
def read_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        current_id = None
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:]  # 获取 ID
                sequences[current_id] = ''
            else:
                sequences[current_id] += line
    return sequences


# Update the preprocess_sequence function to handle batches
def preprocess_sequences(fasta_sequences, max_length=100):
    processed_sequences = torch.zeros((len(fasta_sequences), max_length), dtype=torch.long)
    for i, (sequence_id, sequence) in enumerate(fasta_sequences.items()):
        sequence_indices = [amino_to_index.get(amino, 0) for amino in sequence]
        sequence_indices += [0] * (max_length - len(sequence_indices))
        sequence_indices = sequence_indices[:max_length]
        processed_sequences[i, :] = torch.tensor(sequence_indices)
    return processed_sequences
# Read sequences from a FASTA file
fasta_file = "Zea_mays_rename.orf"
sequences = read_fasta(fasta_file)

# Output file path
output_file = "Zea_mays_rename_prediction.txt"

# Batch processing of sequences
batch_size = 32
sequence_items = list(sequences.items())
sequence_batches = [sequence_items[i:i + batch_size] for i in range(0, len(sequence_items), batch_size)]

# Predict sequences in batches and write to file
with open(output_file, 'w') as file:
    for batch in sequence_batches:
        batch_sequences = {sequence_id: sequence for sequence_id, sequence in batch}
        processed_sequences = preprocess_sequences(batch_sequences, max_length=100)
        with torch.no_grad():
            model_input = processed_sequences.to(torch.device('cuda'))
            output = model(model_input)
            _, predicted_classes = torch.max(output, 1)

        for (sequence_id, _), predicted_class in zip(batch, predicted_classes.tolist()):
            if predicted_class == 1:
                file.write(f"{sequence_id}\n")

print(f"Predictions written to {output_file}")
