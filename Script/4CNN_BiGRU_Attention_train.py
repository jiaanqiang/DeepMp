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
import argparse

def parse_arguments():
    """
    Parse command line arguments for training configuration
    """
    parser = argparse.ArgumentParser(description='Protein Sequence Classification with CNN-BiGRU-Attention')
    
    parser.add_argument('-input', '--input_file', type=str, required=True,
                       help='Path to input CSV file containing protein sequences and labels')
    parser.add_argument('-epochs', '--num_epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('-batch', '--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00001,
                       help='Initial learning rate (default: 0.00001)')
    parser.add_argument('-maxlen', '--max_length', type=int, default=100,
                       help='Maximum sequence length (default: 100)')
    parser.add_argument('-seed', '--random_seed', type=int, default=100,
                       help='Random seed for reproducibility (default: 100)')
    parser.add_argument('-save', '--save_model', type=str, default='best_CNN_gpu.pth',
                       help='Filename for saving best model (default: best_CNN_gpu.pth)')
    parser.add_argument('-device', '--device_id', type=str, default='0',
                       help='CUDA device ID (default: 0)')
    parser.add_argument('-test', '--test_size', type=float, default=0.2,
                       help='Test set size proportion (default: 0.2)')
    
    return parser.parse_args()

# Set CUDA device
args = parse_arguments()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

def set_seed(seed):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.random_seed)

# Amino acid to index mapping (1-20)
amino_to_index = {amino: i+1 for i, amino in enumerate('ARNDCQEGHILKMFPSTWYV')}

class ProteinDataset(Dataset):
    """
    Custom Dataset class for protein sequences
    """
    def __init__(self, data, amino_to_index, max_length=100):
        self.data = data
        self.amino_to_index = amino_to_index
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # Convert amino acid sequence to index sequence and pad/truncate to max_length
        sequence_indices = [self.amino_to_index.get(amino, 0) for amino in sequence]
        sequence_indices += [0] * (self.max_length - len(sequence_indices))
        sequence_indices = sequence_indices[:self.max_length]

        return {
            'sequence_indices': torch.tensor(sequence_indices),
            'label': torch.tensor(label)
        }

class ProteinPredictor(nn.Module):
    """
    CNN-BiGRU-Attention model for protein sequence classification
    """
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
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model weights using Kaiming normal initialization
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.Embedding)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        """
        Forward pass through the network
        """
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)  # Reshape for convolutional layer
        
        # Apply CNN layers
        x = self.conv1(embedded)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)  # Additional convolutional layer
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)  # Additional convolutional layer
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

def main():
    """
    Main training function
    """
    # Print configuration information
    print("=" * 60)
    print("Protein Sequence Classification Training")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Random seed: {args.random_seed}")
    print(f"Test size: {args.test_size}")
    print("=" * 60)
    
    # Load data from CSV file
    print(f"Loading data from {args.input_file}...")
    data = pd.read_csv(args.input_file)
    print(f"Data loaded: {len(data)} samples")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Separate features (sequences) and labels
    X = data.iloc[:, 0]  # First column is sequences
    y = data.iloc[:, 1]  # Second column is labels
    
    print(f"Positive samples: {sum(y == 1)}")
    print(f"Negative samples: {sum(y == 0)}")
    
    # Perform stratified split to maintain class proportions
    train_X_1, test_X_1, train_y_1, test_y_1 = train_test_split(
        X[y == 1], y[y == 1], test_size=args.test_size, random_state=args.random_seed
    )
    train_X_0, test_X_0, train_y_0, test_y_0 = train_test_split(
        X[y == 0], y[y == 0], test_size=args.test_size, random_state=args.random_seed
    )
    
    train_X = pd.concat([train_X_1, train_X_0])
    test_X = pd.concat([test_X_1, test_X_0])
    train_y = pd.concat([train_y_1, train_y_0])
    test_y = pd.concat([test_y_1, test_y_0])
    
    # Combine training and test sets, then shuffle within each class
    train_data = shuffle(pd.DataFrame({'sequence': train_X, 'label': train_y}), 
                        random_state=args.random_seed)
    test_data = shuffle(pd.DataFrame({'sequence': test_X, 'label': test_y}), 
                       random_state=args.random_seed)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create dataset objects
    train_dataset = ProteinDataset(train_data, amino_to_index=amino_to_index, 
                                 max_length=args.max_length)
    test_dataset = ProteinDataset(test_data, amino_to_index=amino_to_index, 
                                max_length=args.max_length)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False)
    
    # Initialize model
    input_size = len(amino_to_index) + 1
    num_classes = 2
    
    model = ProteinPredictor(input_size, num_classes)
    model.to(device)
    
    # Print model information
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Learning rate scheduler parameters
    lr_decay_factor = 0.5
    lr_patience = 4
    current_lr = args.learning_rate
    no_improvement_count = 0
    
    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=1e-5)
    
    # Calculate class weights for imbalanced data
    class_counts = train_data['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weight = torch.tensor(class_weights, dtype=torch.float32)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight.to(device))
    
    # Define learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_factor, 
                                    patience=lr_patience, verbose=True)
    
    # Training and testing loop
    best_roc_auc = 0.0
    history = {
        'train_loss': [],
        'test_accuracy': [],
        'roc_auc': [],
        'sn': [],
        'sp': [],
        'mcc': [],
        'precision': [],
        'f1_score': []
    }
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        
        # Training phase
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
        
        # Validation phase
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
        
        # Apply threshold for binary classification
        threshold = 0.5
        predicted_labels_adjusted = (np.array(all_scores) >= threshold).astype(int)
        
        # Calculate AUC-ROC
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        roc_auc = roc_auc_score(all_labels, all_scores)
        
        # Calculate other evaluation metrics
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
        
        # Save history for tracking
        history['train_loss'].append(avg_loss)
        history['test_accuracy'].append(accuracy)
        history['roc_auc'].append(roc_auc)
        history['sn'].append(sn)
        history['sp'].append(sp)
        history['mcc'].append(mcc)
        history['precision'].append(precision)
        history['f1_score'].append(f1_score)
        
        # Print progress
        print(f'Epoch [{epoch+1:3d}/{args.num_epochs}] | Loss: {avg_loss:.4f} | '
              f'Acc: {accuracy*100:6.2f}% | AUC: {roc_auc:.4f} | Sn: {sn:.4f} | '
              f'Sp: {sp:.4f} | MCC: {mcc:.4f} | F1: {f1_score:.4f}')
        
        # Save model with best AUC-ROC
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            no_improvement_count = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_roc_auc': best_roc_auc,
                'history': history,
                'args': vars(args)
            }, args.save_model)
            print(f'  ✓ Best model saved (AUC: {roc_auc:.4f})')
        else:
            no_improvement_count += 1
            if no_improvement_count >= lr_patience:
                current_lr *= lr_decay_factor
                optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr)
                lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_factor, 
                                                patience=lr_patience, verbose=True)
                print(f'  ↳ Learning rate reduced to {current_lr:.2e}')
        
        # Update learning rate
        lr_scheduler.step(roc_auc)
    
    # Training completed
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best ROC-AUC: {best_roc_auc:.4f}")
    print(f"Model saved as: {args.save_model}")
    print("=" * 60)
    
    # Print final training history summary
    print("\nTraining History Summary:")
    print("-" * 40)
    print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Test Accuracy: {history['test_accuracy'][-1]*100:.2f}%")
    print(f"Final ROC-AUC: {history['roc_auc'][-1]:.4f}")
    print(f"Final Sensitivity: {history['sn'][-1]:.4f}")
    print(f"Final Specificity: {history['sp'][-1]:.4f}")
    print(f"Final MCC: {history['mcc'][-1]:.4f}")
    print(f"Final F1-Score: {history['f1_score'][-1]:.4f}")

if __name__ == "__main__":
    main()
