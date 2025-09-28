import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

# 从本地CSV文件中读取数据
data = pd.read_csv('At_Rice_all_1vs100.csv')

device = torch.device('cuda')

# 分离特征（序列）和标签
X = data.iloc[:, 0]  # 第一列是序列
y = data.iloc[:, 1]  # 第二列是标签

# 进行分层划分，确保两个类别的比例在训练集和测试集中保持一致
train_X_1, test_X_1, train_y_1, test_y_1 = train_test_split(X[y == 1], y[y == 1], test_size=0.2, random_state=100)
train_X_0, test_X_0, train_y_0, test_y_0 = train_test_split(X[y == 0], y[y == 0], test_size=0.2, random_state=100)

train_X = pd.concat([train_X_1, train_X_0])
test_X = pd.concat([test_X_1, test_X_0])
train_y = pd.concat([train_y_1, train_y_0])
test_y = pd.concat([test_y_1, test_y_0])

# 组合训练集和测试集的特征和标签，然后在每个类别内部对数据进行随机洗牌，增加数据随机性
train_data = shuffle(pd.DataFrame({'sequence': train_X, 'label': train_y}), random_state=100)
test_data = shuffle(pd.DataFrame({'sequence': test_X, 'label': test_y}), random_state=100)

# 创建训练集和测试集的数据集对象
train_dataset = ProteinDataset(train_data, amino_to_index=amino_to_index, max_length=100)
test_dataset = ProteinDataset(test_data, amino_to_index=amino_to_index, max_length=100)

# 创建数据加载器
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 定义模型


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

# 初始化模型
input_size = len(amino_to_index) + 1
num_classes = 2


model = ProteinPredictor(input_size, num_classes)
model.to(device)  # Move the model to GPU if available


# 定义学习率调度器参数
initial_lr = 0.00001  # 初始学习率，您可以根据需要调整
lr_decay_factor = 0.5
lr_patience = 4
current_lr = initial_lr
no_improvement_count = 0

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)

class_weight = torch.tensor([0.008, 1.0])  # 根据类别不平衡情况调整权重
criterion = torch.nn.CrossEntropyLoss(weight=class_weight.to(device))

model.to(device)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, matthews_corrcoef, confusion_matrix

# 定义学习率调度器
lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_factor, patience=lr_patience, verbose=True)

# 训练和测试循环
num_epochs = 30
best_roc_auc = 0.0  # Initialize best sensitivity

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_dataloader:
        sequence_indices = batch['sequence_indices'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(sequence_indices)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    # 验证循环
    model.eval()
    true_labels = []
    predicted_labels = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in test_dataloader:
            sequence_indices = batch['sequence_indices'].to(device)
            labels = batch['label'].to(device)

            outputs = model(sequence_indices)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probabilities.cpu().numpy())

    # 调整阈值
    threshold = 0.5  # 你可以根据需要调整这个阈值

    predicted_labels_adjusted = (np.array(all_scores) >= threshold).astype(int)
    # 计算 AUC-ROC 曲线
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = roc_auc_score(all_labels, all_scores)

    # 计算其他评估指标
    mcc = matthews_corrcoef(true_labels, predicted_labels_adjusted)
    conf_matrix = confusion_matrix(true_labels, predicted_labels_adjusted)

    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]

    sn = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    f1_score = 2 * (precision * sn) / (precision + sn) if (precision + sn) != 0 else 0.0

    print('Epoch [{}/{}]\tAvg Loss: {:.4f}, True Positives: {:.4f}, True Negatives: {:.4f}, False Positives: {:.4f}, False Negatives: {:.4f},'
        .format(epoch+1, num_epochs, avg_loss, tp, tn, fp, fn))

    print('Epoch [{}/{}]\tTrain Loss: {:.4f}\tTest Accuracy: {:.2f}%, AUC: {:.4f}, Sn: {:.4f}, Sp: {:.4f}, Mcc: {:.4f}, Precision: {:.4f}, f1_score: {:.4f}'
        .format(epoch+1, num_epochs, avg_loss, accuracy * 100, roc_auc, sn, sp, mcc, precision, f1_score))
    
    # 绘制 AUC-ROC 曲线
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_epoch_{epoch+1}.pdf')  # 保存为 PDF 文件
    plt.close()
    # 保存在sn最好时的模型
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        no_improvement_count = 0  # Reset the counter when accuracy improves
        # 保存模型检查点
        torch.save(model.state_dict(), 'best_CNN_gpu.pth')
    else:
        no_improvement_count += 1
        if no_improvement_count >= lr_patience:
            current_lr *= lr_decay_factor
            optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr)
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_factor, patience=lr_patience, verbose=True)
            print(f'Learning rate reduced to {current_lr}')

    # 更新学习率
    lr_scheduler.step(roc_auc)
