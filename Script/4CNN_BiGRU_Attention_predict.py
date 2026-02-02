import os
import torch
import torch.nn as nn
import pandas as pd
import argparse

# Amino acid to index mapping (1-20)
amino_to_index = {amino: i+1 for i, amino in enumerate('ARNDCQEGHILKMFPSTWYV')}

# Define model
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

# Load FASTA file
def read_fasta(file_path):
    """Read sequences from a FASTA file"""
    sequences = {}
    with open(file_path, 'r') as file:
        current_id = None
        current_seq = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]  # Take only the ID part before space
                current_seq = []
            else:
                current_seq.append(line)
        # Don't forget the last sequence
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    return sequences

# Update the preprocess_sequence function to handle batches
def preprocess_sequences(fasta_sequences, max_length=100):
    """Preprocess sequences for model input"""
    processed_sequences = torch.zeros((len(fasta_sequences), max_length), dtype=torch.long)
    for i, (sequence_id, sequence) in enumerate(fasta_sequences.items()):
        sequence_indices = [amino_to_index.get(amino, 0) for amino in sequence]
        sequence_indices += [0] * (max_length - len(sequence_indices))
        sequence_indices = sequence_indices[:max_length]
        processed_sequences[i, :] = torch.tensor(sequence_indices)
    return processed_sequences

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict protein sequences using pre-trained model')
    parser.add_argument('-input', '--input', type=str, required=True,
                       help='Path to input FASTA file (e.g., Test.fasta)')
    parser.add_argument('-output', '--output', type=str, default='prediction_results.txt',
                       help='Path to output prediction file (default: prediction_results.txt)')
    parser.add_argument('-model', '--model_path', type=str, default='best_CNN_gpu.pth',
                       help='Path to pre-trained model (default: best_CNN_gpu.pth)')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=32,
                       help='Batch size for prediction (default: 32)')
    parser.add_argument('-max_length', '--max_length', type=int, default=100,
                       help='Maximum sequence length (default: 100)')
    parser.add_argument('-gpu', '--gpu_id', type=str, default='0',
                       help='GPU ID to use (default: 0)')
    parser.add_argument('-threshold', '--threshold', type=float, default=0.5,
                       help='Probability threshold for positive prediction (default: 0.5)')
    parser.add_argument('--save_all', action='store_true',
                       help='Save all predictions with probabilities, not just positive ones')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        return
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input FASTA file '{args.input}' not found!")
        return
    
    # Output configuration
    print("=" * 50)
    print("Prediction Configuration:")
    print(f"  Input FASTA file: {args.input}")
    print(f"  Output file: {args.output}")
    print(f"  Model file: {args.model_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Probability threshold: {args.threshold}")
    print(f"  Save all predictions: {args.save_all}")
    print("=" * 50)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ProteinPredictor(input_size=len(amino_to_index) + 1, num_classes=2)
    model.to(device)
    
    # Load pre-trained model
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    # Read sequences from FASTA file
    try:
        sequences = read_fasta(args.input)
        print(f"Successfully loaded {len(sequences)} sequences from {args.input}")
        
        if len(sequences) == 0:
            print("Error: No sequences found in the input file!")
            return
            
        # Show some sample sequences
        print("\nSample sequences (first 3):")
        for i, (seq_id, seq) in enumerate(list(sequences.items())[:3]):
            print(f"  {seq_id}: {seq[:50]}... (length: {len(seq)})")
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return
    
    # Batch processing
    batch_size = args.batch_size
    sequence_items = list(sequences.items())
    sequence_batches = [sequence_items[i:i + batch_size] for i in range(0, len(sequence_items), batch_size)]
    
    print(f"\nProcessing {len(sequences)} sequences in {len(sequence_batches)} batches...")
    
    # Predict sequences in batches
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(sequence_batches):
            batch_sequences = {sequence_id: sequence for sequence_id, sequence in batch}
            processed_sequences = preprocess_sequences(batch_sequences, max_length=args.max_length)
            
            model_input = processed_sequences.to(device)
            output = model(model_input)
            
            # Get probabilities
            probabilities = torch.softmax(output, dim=1)
            positive_probs = probabilities[:, 1].cpu().numpy()
            
            # Get predicted classes
            _, predicted_classes = torch.max(output, 1)
            predicted_classes = predicted_classes.cpu().numpy()
            
            # Store predictions
            for (sequence_id, sequence), pred_class, pos_prob in zip(batch, predicted_classes, positive_probs):
                all_predictions.append({
                    'sequence_id': sequence_id,
                    'length': len(sequence),
                    'predicted_class': int(pred_class),
                    'positive_probability': float(pos_prob),
                    'is_positive': pos_prob >= args.threshold
                })
            
            # Progress update
            if (batch_idx + 1) % max(1, len(sequence_batches) // 10) == 0 or (batch_idx + 1) == len(sequence_batches):
                print(f"  Processed batch {batch_idx + 1}/{len(sequence_batches)}")
    
    # Write results to file
    try:
        with open(args.output, 'w') as file:
            if args.save_all:
                # Save all predictions with details
                file.write("Sequence_ID\tLength\tPredicted_Class\tPositive_Probability\tIs_Positive\n")
                for pred in all_predictions:
                    file.write(f"{pred['sequence_id']}\t{pred['length']}\t{pred['predicted_class']}\t{pred['positive_probability']:.4f}\t{pred['is_positive']}\n")
                print(f"\nSaved all predictions to {args.output}")
            else:
                # Save only positive predictions (original behavior)
                positive_count = 0
                for pred in all_predictions:
                    if pred['is_positive']:
                        file.write(f"{pred['sequence_id']}\n")
                        positive_count += 1
                print(f"\nSaved {positive_count} positive predictions to {args.output}")
        
        # Summary statistics
        positive_count = sum(1 for pred in all_predictions if pred['is_positive'])
        avg_positive_prob = sum(pred['positive_probability'] for pred in all_predictions if pred['is_positive']) / max(1, positive_count)
        
        print("\n" + "=" * 50)
        print("Prediction Summary:")
        print(f"  Total sequences: {len(all_predictions)}")
        print(f"  Positive predictions: {positive_count} ({positive_count/len(all_predictions)*100:.1f}%)")
        print(f"  Negative predictions: {len(all_predictions) - positive_count}")
        print(f"  Average positive probability: {avg_positive_prob:.4f}")
        
        # Show some predictions
        print("\nSample predictions (first 5):")
        for pred in all_predictions[:5]:
            status = "POSITIVE" if pred['is_positive'] else "negative"
            print(f"  {pred['sequence_id']}: {status} (prob: {pred['positive_probability']:.4f})")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        return

if __name__ == "__main__":
    main()
