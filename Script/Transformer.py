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

seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Mapping of Amino Acids to Indices (1-20)
amino_to_index = {amino: i+1 for i, amino in enumerate('ARNDCQEGHILKMFPSTWYV')}

# Custom dataset class
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

        # Convert amino acid sequence to index sequence
        sequence_indices = [self.amino_to_index.get(amino, 0) for amino in sequence]
        sequence_indices += [0] * (self.max_length - len(sequence_indices))
        sequence_indices = sequence_indices[:self.max_length]

        return {
            'sequence_indices': torch.tensor(sequence_indices),
            'label': torch.tensor(label)
        }

# Read data from a local CSV file
data = pd.read_csv('Data_1vs1.csv')

device = torch.device('cuda')
# Separating features (sequences) and labels
X = data.iloc[:, 0]  # The first column is the sequence
y = data.iloc[:, 1]  # The second column is the label

# Split the data ensuring consistent class ratios in train and test sets
train_X_1, test_X_1, train_y_1, test_y_1 = train_test_split(X[y == 1], y[y == 1], test_size=0.3, random_state=100)
train_X_0, test_X_0, train_y_0, test_y_0 = train_test_split(X[y == 0], y[y == 0], test_size=0.3, random_state=100)

train_X = pd.concat([train_X_1, train_X_0])
test_X = pd.concat([test_X_1, test_X_0])
train_y = pd.concat([train_y_1, train_y_0])
test_y = pd.concat([test_y_1, test_y_0])

# Combine features and labels for train and test sets and shuffle the data within each class to add randomness
train_data = shuffle(pd.DataFrame({'sequence': train_X, 'label': train_y}), random_state=100)
test_data = shuffle(pd.DataFrame({'sequence': test_X, 'label': test_y}), random_state=100)

# Create dataset objects for train and test sets
train_dataset = ProteinDataset(train_data, amino_to_index=amino_to_index, max_length=100)
test_dataset = ProteinDataset(test_data, amino_to_index=amino_to_index, max_length=100)

# Create data loaders
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define model
class ProteinPredictor(nn.Module):
    def __init__(self, num_labels, max_length=100):
        super(ProteinPredictor, self).__init__()
        self.embedding = nn.Embedding(len(amino_to_index) + 1, 128)
        self.transformer = nn.Transformer(d_model=128, nhead=8, num_encoder_layers=4)
        self.fc = nn.Linear(max_length * 128, num_labels)

    def forward(self, sequence_indices):
        embedded = self.embedding(sequence_indices)
        transformer_output = self.transformer(embedded.permute(1, 0, 2), embedded.permute(1, 0, 2))
        pooled_output = transformer_output.permute(1, 0, 2).reshape(sequence_indices.size(0), -1)
        logits = self.fc(pooled_output)
        return logits


num_labels = 2
model = ProteinPredictor(num_labels, max_length=100)
model.to(device)  # Move the model to GPU if available

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

#  Define learning rate scheduler parameters
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5, verbose=True)


# Training and testing loops
num_epochs = 30
best_accuracy = 0.0
all_labels = []
all_scores = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        sequence_indices = batch['sequence_indices'].to(device)  # Move data to GPU
        labels = batch['label'].to(device)  # Move data to GPU

        optimizer.zero_grad()
        outputs = model(sequence_indices)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)

    # Validation Loop
    model.eval()
    true_labels = []  # Initialize true_labels and predicted_labels lists
    predicted_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            sequence_indices = batch['sequence_indices'].to(device)  # Move data to GPU
            labels = batch['label'].to(device)  # Move data to GPU

            outputs = model(sequence_indices)
            _, predicted = torch.max(outputs.data, 1)
            
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1] 

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probabilities.cpu().numpy())

        auc = roc_auc_score(all_labels, all_scores)

        mcc = matthews_corrcoef(true_labels, predicted_labels)

        conf_matrix = confusion_matrix(true_labels, predicted_labels)



        #precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0

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
            .format(epoch+1, num_epochs, avg_loss, accuracy * 100, auc, sn, sp, mcc, precision,f1_score))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save the model checkpoint
            torch.save(model.state_dict(), 'best_model_Trans.pth')
    # Update the learning rate scheduler
    scheduler.step(accuracy)
