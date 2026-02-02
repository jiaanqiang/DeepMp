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
import argparse

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Amino acid to index mapping (1-20)
amino_to_index = {amino: i+1 for i, amino in enumerate('ARNDCQEGHILKMFPSTWYV')}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Protein Sequence Prediction using CNN-BiGRU-Attention Model')
    
    parser.add_argument('-input', '--input_file', type=str, required=True,
                       help='Path to input FASTA file containing protein sequences')
    parser.add_argument('-output', '--output_file', type=str, required=True,
                       help='Path to output file for prediction results')
    parser.add_argument('-model', '--model_path', type=str, default='best_CNN_gpu.pth',
                       help='Path to trained model checkpoint (default: best_CNN_gpu.pth)')
    parser.add_argument('-batch', '--batch_size', type=int, default=32,
                       help='Batch size for prediction (default: 32)')
    parser.add_argument('-maxlen', '--max_length', type=int, default=100,
                       help='Maximum sequence length (default: 100)')
    parser.add_argument('-device', '--device_id', type=str, default='0',
                       help='CUDA device ID (default: 0)')
    parser.add_argument('-threshold', '--pred_threshold', type=float, default=0.5,
                       help='Prediction threshold for positive class (default: 0.5)')
    parser.add_argument('-save_prob', '--save_probabilities', action='store_true',
                       help='Save prediction probabilities along with class labels')
    
    return parser.parse_args()

# Protein predictor model definition
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
        """Initialize model weights using Kaiming normal initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.Embedding)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        """Forward pass through the model"""
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

def read_fasta(file_path):
    """
    Read sequences from a FASTA file
    
    Args:
        file_path: Path to the FASTA file
    
    Returns:
        Dictionary with sequence IDs as keys and sequences as values
    """
    sequences = {}
    with open(file_path, 'r') as file:
        current_id = None
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:]  # Get ID (remove '>')
                sequences[current_id] = ''
            else:
                sequences[current_id] += line.upper()  # Convert to uppercase
    return sequences

def preprocess_sequences(fasta_sequences, max_length=100):
    """
    Preprocess sequences for model input
    
    Args:
        fasta_sequences: Dictionary of sequences
        max_length: Maximum sequence length for padding/truncation
    
    Returns:
        Tensor of processed sequences
    """
    processed_sequences = torch.zeros((len(fasta_sequences), max_length), dtype=torch.long)
    for i, (sequence_id, sequence) in enumerate(fasta_sequences.items()):
        # Convert amino acids to indices
        sequence_indices = [amino_to_index.get(amino, 0) for amino in sequence]
        
        # Pad or truncate to max_length
        if len(sequence_indices) < max_length:
            sequence_indices += [0] * (max_length - len(sequence_indices))
        else:
            sequence_indices = sequence_indices[:max_length]
        
        processed_sequences[i, :] = torch.tensor(sequence_indices)
    
    return processed_sequences

def validate_sequences(sequences):
    """
    Validate that sequences contain only valid amino acids
    
    Args:
        sequences: Dictionary of sequences
    
    Returns:
        List of sequence IDs with invalid sequences
    """
    valid_amino_acids = set('ARNDCQEGHILKMFPSTWYV')
    invalid_sequences = []
    
    for seq_id, sequence in sequences.items():
        if not set(sequence).issubset(valid_amino_acids):
            invalid_sequences.append(seq_id)
    
    return invalid_sequences

def main():
    """Main function for prediction"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print configuration
    print("=" * 60)
    print("Protein Sequence Prediction")
    print("=" * 60)
    print(f"Input FASTA file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model checkpoint: {args.model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Prediction threshold: {args.pred_threshold}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Read sequences from FASTA file
    print(f"Reading sequences from {args.input_file}...")
    sequences = read_fasta(args.input_file)
    print(f"Loaded {len(sequences)} sequences")
    
    # Validate sequences
    invalid_seqs = validate_sequences(sequences)
    if invalid_seqs:
        print(f"Warning: {len(invalid_seqs)} sequences contain invalid amino acids")
        if len(invalid_seqs) > 5:
            print(f"First 5 invalid sequence IDs: {invalid_seqs[:5]}")
        else:
            print(f"Invalid sequence IDs: {invalid_seqs}")
    
    # Load the pre-trained model
    print("Loading model...")
    model = ProteinPredictor(input_size=len(amino_to_index) + 1, num_classes=2)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Batch processing of sequences
    batch_size = args.batch_size
    sequence_items = list(sequences.items())
    sequence_batches = [sequence_items[i:i + batch_size] 
                       for i in range(0, len(sequence_items), batch_size)]
    
    print(f"Processing {len(sequence_batches)} batch(es) of up to {batch_size} sequences each...")
    
    # Open output file
    with open(args.output_file, 'w') as file:
        # Write header
        if args.save_probabilities:
            file.write("# Sequence_ID\n")
        else:
            file.write("# Sequence_ID\n")
        
        total_positive = 0
        total_negative = 0
        
        for batch_idx, batch in enumerate(sequence_batches, 1):
            batch_sequences = {sequence_id: sequence for sequence_id, sequence in batch}
            
            try:
                # Preprocess sequences
                processed_sequences = preprocess_sequences(batch_sequences, max_length=args.max_length)
                
                # Make predictions
                with torch.no_grad():
                    model_input = processed_sequences.to(device)
                    output = model(model_input)
                    
                    # Get probabilities and predicted classes
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    positive_probs = probabilities[:, 1].cpu().numpy()
                    
                    # Apply threshold
                    predicted_classes = (positive_probs >= args.pred_threshold).astype(int)
                
                # Write results to file
                for (sequence_id, _), pred_class, prob in zip(batch, predicted_classes, positive_probs):
                    if args.save_probabilities:
                        file.write(f"{sequence_id}\t{pred_class}\t{prob:.4f}\n")
                    else:
                        if pred_class == 1:
                            file.write(f"{sequence_id}\n")
                    
                    if pred_class == 1:
                        total_positive += 1
                    else:
                        total_negative += 1
                
                print(f"Processed batch {batch_idx}/{len(sequence_batches)} "
                      f"({len(batch)} sequences)")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("Prediction Summary:")
    print("=" * 60)
    print(f"Total sequences processed: {len(sequences)}")
    print(f"Positive predictions (Class 1): {total_positive}")
    print(f"Negative predictions (Class 0): {total_negative}")
    print(f"Positive rate: {(total_positive/len(sequences)*100):.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("=" * 60)
    
    # Save detailed statistics if requested
    if args.save_probabilities:
        stats_file = args.output_file.replace('.txt', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("Prediction Statistics\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total sequences: {len(sequences)}\n")
            f.write(f"Positive predictions: {total_positive}\n")
            f.write(f"Negative predictions: {total_negative}\n")
            f.write(f"Positive rate: {(total_positive/len(sequences)*100):.2f}%\n")
            f.write(f"Threshold used: {args.pred_threshold}\n")
        print(f"Detailed statistics saved to: {stats_file}")

if __name__ == "__main__":
    main()
