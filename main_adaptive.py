import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Function to load data from folders
def load_data_from_folders(folder_paths):
    all_texts = []
    all_labels = []
    all_prompts = []
    all_ids = []
    
    for label, folder_path in enumerate(folder_paths):
        texts = []
        prompts = []
        ids = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as file:
                    for line in file:
                        try:
                            data = json.loads(line.strip())
                            if 'textwm' in data:  # Using the watermarked text
                                texts.append(data['textwm'])
                                # Store prompt and idx if available
                                prompts.append(data.get('prompt', ''))
                                ids.append(data.get('idx', str(len(ids))))
                        except json.JSONDecodeError:
                            continue
        
        all_texts.extend(texts)
        all_labels.extend([label] * len(texts))
        all_prompts.extend(prompts)
        all_ids.extend(ids)
    
    return all_texts, all_labels, all_prompts, all_ids

# Function to extract key from folder path
def extract_key_from_path(folder_path):
    # Split by dash and get the last element
    if '-' in folder_path:
        return folder_path.split('-')[-1]
    else:
        # Handle the special case for selfhash
        return '15485863' if 'selfhash' in folder_path else 'unknown'

# Text classifier model
class TextClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(TextClassifier, self).__init__()
        self.bert = pretrained_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
# Modified WatermarkDataset class
class WatermarkDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, include_original_text=False, 
                 original_indices=None, prompts=None, ids=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_original_text = include_original_text
        self.original_indices = original_indices or list(range(len(texts)))
        self.prompts = prompts or [''] * len(texts)
        self.ids = ids or [str(i) for i in range(len(texts))]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        original_idx = self.original_indices[idx]
        prompt = self.prompts[idx]
        id_val = self.ids[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long),
            'original_idx': original_idx,
            'prompt': prompt,
            'idx': id_val
        }
        
        if self.include_original_text:
            item['text'] = text
            
        return item

# Training function
def train_model(model, train_dataloader, val_dataloader, device, num_epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    best_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                _, preds = torch.max(outputs, 1)
                
                val_preds.extend(preds.cpu().tolist())
                val_true.extend(labels.cpu().tolist())
        
        val_accuracy = accuracy_score(val_true, val_preds)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg. Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_accuracy

# Evaluation function
def evaluate_model(model, test_dataloader, device):
    model.eval()
    test_preds = []
    test_true = []
    test_indices = []
    test_texts = []
    test_prompts = []
    test_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            indices = batch['original_idx']
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().tolist())
            test_true.extend(labels.cpu().tolist())
            test_indices.extend(indices.cpu().tolist())
            
            # Store the full texts, prompts and ids
            if 'text' in batch:
                test_texts.extend(batch['text'])
            if 'prompt' in batch:
                test_prompts.extend(batch['prompt'])
            if 'idx' in batch:
                test_ids.extend(batch['idx'])
    
    test_accuracy = accuracy_score(test_true, test_preds)
    
    # Return predictions and all relevant data
    return test_accuracy, test_preds, test_true, test_indices, test_texts, test_prompts, test_ids

# Updated export function to include idx, prompt, and textwm
def export_classified_data(texts, true, predictions, keys, sample_size, prompts, ids, output_dir="out_adaptive_attacker"):
    # Create counters for each key to track number of samples
    key_counters = {key: 0 for key in keys}
    
    # Create directories for each key
    base_dir = os.path.join(output_dir, str(sample_size), "ours")
    os.makedirs(base_dir, exist_ok=True)
    
    # Initialize files for each key
    key_files = {}
    for i, key in enumerate(keys):
        if key != "selfhash":
            key_dir = os.path.join(base_dir, f"c4-kgw-ff-anchored_minhash_prf-4-True-{key}") # Example key directory
        else:
            key_dir = os.path.join(base_dir, f"c4-kgw-ff-anchored_minhash_prf-4-True-15485863")
        os.makedirs(key_dir, exist_ok=True)
        filename = os.path.join(key_dir, "0.jsonl")
        key_files[i] = open(filename, 'w')
        print(f"Writing to {filename}")
    
    # Write data to files based on predictions
    for i, (text, true_t, pred, prompt, id_val) in enumerate(zip(texts, true, predictions, prompts, ids)):
        if pred < len(keys) and pred == true_t:  # Check if prediction is valid
            json_data = json.dumps({
                "idx": id_val,
                "prompt": prompt,
                "textwm": text
            })
            # "predicted_key_idx": int(pred)
            key_files[pred].write(json_data + '\n')
            key_counters[keys[pred]] += 1
    
    # Close all files
    for file in key_files.values():
        file.close()
    
    # Print statistics
    print("\nClassification statistics:")
    for key, count in key_counters.items():
        print(f"Key {key}: {count} samples")
    
    return key_counters

# Modified main experiment function
def run_key_clustering_experiment(folder_paths, sample_sizes, num_test_samples=10000, batch_size=16):
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Extract keys from folder paths
    keys = [extract_key_from_path(path) for path in folder_paths]
    print(f"Detected keys: {keys}")
    
    # Load all data
    print("Loading data from folders...")
    all_texts, all_labels, all_prompts, all_ids = load_data_from_folders(folder_paths)
    
    # Make sure all lists have the same length
    assert len(all_texts) == len(all_labels) == len(all_prompts) == len(all_ids), "Data lists have different lengths"
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"  # Smaller model for faster training
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Combine data into a single list of tuples for splitting
    combined_data = list(zip(all_texts, all_labels, all_prompts, all_ids))
    
    # Split into train and test sets
    train_data, test_data = train_test_split(
        combined_data, test_size=num_test_samples, 
        stratify=[d[1] for d in combined_data],  # Stratify by labels
        random_state=42
    )
    
    # Unpack the train and test data
    train_texts, train_labels, train_prompts, train_ids = zip(*train_data)
    test_texts, test_labels, test_prompts, test_ids = zip(*test_data)
    
    # Prepare test dataset (constant for all experiments)
    test_dataset = WatermarkDataset(
        test_texts, test_labels, tokenizer, 
        include_original_text=True, 
        original_indices=list(range(len(test_texts))),
        prompts=test_prompts,
        ids=test_ids
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Run experiments for different sample sizes
    results = []
    
    for sample_size in sample_sizes:
        print(f"\n--- Experiment with {sample_size} training samples per class ---")
        
        # Sample balanced data for each class
        sampled_data = []
        
        for class_idx in range(len(folder_paths)):
            # Get all indices for this class
            class_indices = [i for i, label in enumerate(train_labels) if label == class_idx]
            
            if len(class_indices) >= sample_size:
                # Randomly select indices for this class
                selected_indices = random.sample(class_indices, sample_size)
                # Add the corresponding data points
                for idx in selected_indices:
                    sampled_data.append((
                        train_texts[idx], 
                        train_labels[idx],
                        train_prompts[idx],
                        train_ids[idx]
                    ))
            else:
                print(f"Warning: Not enough samples for class {class_idx}. Using all {len(class_indices)} available samples.")
                for idx in class_indices:
                    sampled_data.append((
                        train_texts[idx], 
                        train_labels[idx],
                        train_prompts[idx],
                        train_ids[idx]
                    ))
        
        # Unpack the sampled data
        sampled_texts, sampled_labels, sampled_prompts, sampled_ids = zip(*sampled_data)
        
        # Split into train and validation
        train_split_data, val_split_data = train_test_split(
            list(zip(sampled_texts, sampled_labels, sampled_prompts, sampled_ids)),
            test_size=0.1, 
            stratify=sampled_labels,
            random_state=42
        )
        
        # Unpack train and validation data
        train_texts_split, train_labels_split, train_prompts_split, train_ids_split = zip(*train_split_data)
        val_texts, val_labels, val_prompts, val_ids = zip(*val_split_data)
        
        # Create datasets and dataloaders
        train_dataset = WatermarkDataset(
            train_texts_split, train_labels_split, tokenizer,
            prompts=train_prompts_split, ids=train_ids_split
        )
        val_dataset = WatermarkDataset(
            val_texts, val_labels, tokenizer,
            prompts=val_prompts, ids=val_ids
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize and train model
        base_model = AutoModel.from_pretrained(model_name)
        model = TextClassifier(base_model, num_classes=len(folder_paths))
        model = model.to(device)
        
        model, val_accuracy = train_model(
            model, train_dataloader, val_dataloader, device, num_epochs=3
        )
        
        # Evaluate on test set and get predictions
        test_accuracy, test_preds, test_true, test_indices, test_text_contents, test_prompt_contents, test_id_contents = evaluate_model(
            model, test_dataloader, device
        )
        print(f"Test Accuracy with {sample_size} samples per class: {test_accuracy:.4f}")
        
        # Export the classified test data
        key_counts = export_classified_data(
            test_text_contents, 
            test_true,
            test_preds, 
            keys, 
            sample_size,
            test_prompt_contents,
            test_id_contents
        )
        # texts, true, predictions, keys, sample_size, prompts, ids
        
        results.append((sample_size, test_accuracy))
    
    return results

# Set larger font sizes globally
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def plot_results(output_file="out_adaptive_attacker/key_clustering_results.pdf"):
    sample_sizes = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot clustering accuracy in red
    line1 = ax.plot(sample_sizes, accuracies, marker='o', linestyle='-', linewidth=2.5, 
             color='blue', label='Clustering Accuracy', markersize=10)
    
    # Set axis labels
    ax.set_xlabel('Number of Training Samples per Key', fontweight='bold')
    ax.set_ylabel('Success Rate', fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim([0.3, 1.0])
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    return fig

if __name__ == "__main__":
    # Configuration
    folder_paths = [
        "/out_mistral/ours/c4-kgw-ff-anchored_minhash_prf-4-True-15485864",
        "/out_mistral/ours/c4-kgw-ff-anchored_minhash_prf-4-True-15485865",
        "/out_mistral/ours/c4-kgw-ff-anchored_minhash_prf-4-True-15485866",
        "/out_mistral/ours/c4-kgw-selfhash",
    ]
    
    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    
    # Run experiment
    results = run_key_clustering_experiment(folder_paths, sample_sizes, num_test_samples=10000, batch_size=16)
    
    # Plot results
    plot_results(results)