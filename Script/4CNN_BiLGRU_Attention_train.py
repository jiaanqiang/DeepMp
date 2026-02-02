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
import argparse  # 添加 argparse 模块

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='Train Protein Predictor Model')
    parser.add_argument('-input', '--input', type=str, required=True,
                       help='Path to input CSV file (e.g., Data/Data_1vs1.csv)')
    parser.add_argument('-output', '--output', type=str, default='best_model_CNN.pth',
                       help='Path to save the best model (default: best_model_CNN.pth)')
    parser.add_argument('-epochs', '--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001)')
    parser.add_argument('-max_length', '--max_length', type=int, default=100,
                       help='Maximum sequence length (default: 100)')
    parser.add_argument('-gpu', '--gpu_id', type=str, default='0',
                       help='GPU ID to use (default: 0)')
    
    return parser.parse_args()

# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # 输出参数信息
    print("=" * 50)
    print("Training Configuration:")
    print(f"  Input file: {args.input}")
    print(f"  Output model: {args.output}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  GPU ID: {args.gpu_id}")
    print("=" * 50)
    
    # Mapping of Amino Acids to Indices (1-20)
    amino_to_index = {amino: i+1 for i, amino in enumerate('ARNDCQEGHILKMFPSTWYV')}

    # Custom dataset class
    class ProteinDataset(Dataset):
        def __init__(self, data, amino_to_index, max_length=args.max_length):
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

    # Read data from the input CSV file
    try:
        data = pd.read_csv(args.input)
        print(f"Successfully loaded data from {args.input}")
        print(f"Dataset size: {len(data)} samples")
        print(f"Positive samples: {sum(data.iloc[:, 1] == 1)}")
        print(f"Negative samples: {sum(data.iloc[:, 1] == 0)}")
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found!")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Separating features (sequences) and labels
    X = data.iloc[:, 0]  # The first column is the sequence
    y = data.iloc[:, 1]  # The second column is the label

    # Split the data ensuring consistent class ratios in train and test sets
    train_X_1, test_X_1, train_y_1, test_y_1 = train_test_split(X[y == 1], y[y == 1], test_size=0.2, random_state=100)
    train_X_0, test_X_0, train_y_0, test_y_0 = train_test_split(X[y == 0], y[y == 0], test_size=0.2, random_state=100)

    train_X = pd.concat([train_X_1, train_X_0])
    test_X = pd.concat([test_X_1, test_X_0])
    train_y = pd.concat([train_y_1, train_y_0])
    test_y = pd.concat([test_y_1, test_y_0])

    # Combine features and labels for train and test sets and shuffle the data within each class to add randomness
    train_data = shuffle(pd.DataFrame({'sequence': train_X, 'label': train_y}), random_state=100)
    test_data = shuffle(pd.DataFrame({'sequence': test_X, 'label': test_y}), random_state=100)

    # Create dataset objects for train and test sets
    train_dataset = ProteinDataset(train_data, amino_to_index=amino_to_index, max_length=args.max_length)
    test_dataset = ProteinDataset(test_data, amino_to_index=amino_to_index, max_length=args.max_length)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Define model
    class ProteinPredictor(nn.Module):
        def __init__(self, input_size, num_classes, max_length=args.max_length):
            super(ProteinPredictor, self).__init__()

            embedding_dim = 128
            hidden_size = 128
            
            # Define the layers
            self.embedding = nn.Embedding(input_size, embedding_dim)
            self.conv1 = nn.Conv1d(embedding_dim, 256, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(1024, 2048, kernel_size=3, padding=1)
            self.maxpool = nn.MaxPool1d(kernel_size=2)
            self.dropout = nn.Dropout(0.2)
            # Adjust the input size for GRU to match the output size of convolutions
            self.bigru = nn.GRU(2048, hidden_size, num_layers=6, batch_first=True, bidirectional=True, dropout=0.2)
            self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=8, dropout=0.2)
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
            x = self.conv3(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv4(x)
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

    # Initialize model
    input_size = len(amino_to_index) + 1
    num_classes = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ProteinPredictor(input_size, num_classes)
    model.to(device)  # Move the model to GPU if available

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Define learning rate scheduler parameters
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5, verbose=True)

    # Training and testing loops
    num_epochs = args.epochs
    best_accuracy = 0.0
    all_labels = []
    all_scores = []

    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            sequence_indices = batch['sequence_indices'].to(device)  # Move data to device
            labels = batch['label'].to(device)  # Move data to device

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
                sequence_indices = batch['sequence_indices'].to(device)  # Move data to device
                labels = batch['label'].to(device)  # Move data to device

                outputs = model(sequence_indices)
                _, predicted = torch.max(outputs.data, 1)
                
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1]

                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probabilities.cpu().numpy())

            auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0

            mcc = matthews_corrcoef(true_labels, predicted_labels)

            conf_matrix = confusion_matrix(true_labels, predicted_labels)

            tp = conf_matrix[1, 1]
            tn = conf_matrix[0, 0]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            sn = tp / (tp + fn) if (tp + fn) != 0 else 0.0
            sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            accuracy = (tp + tn) / (tp + tn + fp + fn)

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
            f1_score = 2 * (precision * sn) / (precision + sn) if (precision + sn) != 0 else 0.0

            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'  Avg Loss: {avg_loss:.4f}')
            print(f'  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}')
            print(f'  Metrics - Accuracy: {accuracy*100:.2f}%, AUC: {auc:.4f}, Sn: {sn:.4f}, Sp: {sp:.4f}, Mcc: {mcc:.4f}, Precision: {precision:.4f}, F1: {f1_score:.4f}')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Save the model checkpoint
                torch.save(model.state_dict(), args.output)
                print(f'  ✓ Saved best model to {args.output}')
        
        # Update the learning rate scheduler
        scheduler.step(accuracy)
    
    print("\n" + "=" * 50)
    print(f"Training completed!")
    print(f"Best accuracy: {best_accuracy*100:.2f}%")
    print(f"Model saved to: {args.output}")
    print("=" * 50)

if __name__ == "__main__":
    main()
