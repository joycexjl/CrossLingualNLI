import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from nli_model import CrossLingualNLIModel
import pandas as pd
import json
from transformers import BertTokenizer

class XNLIDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read the file and process each line
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                json_line = json.loads(line)
                self.data.append((json_line['premise'], json_line['hypothesis'], json_line['label']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise, hypothesis, label = self.data[idx]

        # Tokenize premise and hypothesis
        encoded_pair = self.tokenizer.encode_plus(
            premise, hypothesis, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        # Convert label to a tensor
        label_tensor = torch.tensor(label)  # Assuming label is an integer

        return encoded_pair['input_ids'].squeeze(0), encoded_pair['attention_mask'].squeeze(0), label_tensor

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Create datasets
train_dataset = XNLIDataset('multinli_1.0/multinli_1.0_train.jsonl', tokenizer)
dev_dataset = XNLIDataset('XNLI-1.0/xnli.dev.jsonl', tokenizer)
test_dataset = XNLIDataset('XNLI-1.0/xnli.test.jsonl', tokenizer)

# Create DataLoaders
train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_data_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = CrossLingualNLIModel('en', 'zh') # Specify which language the model should be trained ond
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 3 # Define num_epoch

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Compute loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Compute accuracy
            preds = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)
    return avg_loss, accuracy

# Assuming you've set up the model, data loaders, optimizer, and criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    train_loss = train(model, train_data_loader, optimizer, criterion, device)
    val_loss, val_accuracy = evaluate(model, dev_data_loader, criterion, device)
    print(f"Epoch {epoch}: Train Loss = {train_loss}, Validation Loss = {val_loss}, Validation Accuracy = {val_accuracy}")
