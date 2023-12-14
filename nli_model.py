import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import math
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from alignment_model import WordAlignmentModel


from transformers import get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# Load the dataset
df = pd.read_json('multinli_1.0/multinli_1.0_train.jsonl')

# Encode all data
encoded_data = [encode_sentences(tokenizer, row['premise'], row['hypothesis']) for index, row in df.iterrows()]

class XNLIDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Convert encoded data to a dataset
labels = df['label'].tolist() # Assuming labels are stored in a column 'label'
dataset = XNLIDataset(encoded_data, labels)

# Create a DataLoader for batching
train_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Defining the Transformer Model Components

def encode_sentences(tokenizer, premise, hypothesis, max_length=256):
    # Tokenize and encode the sentences
    encoded_pair = tokenizer.encode_plus(
        premise, hypothesis, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    return encoded_pair['input_ids'], encoded_pair['attention_mask']

class BertNLIModel(nn.Module):
    def __init__(self, num_labels=3):  # 3 labels for NLI: entailment, neutral, contradiction
        super(BertNLIModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

num_epochs = 3  # Define the number of epochs

# Instantiate the model
model = BertNLIModel()

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Scheduler for learning rate decay
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_data_loader) * num_epochs)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_data_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs[0]

        # Backward pass and optimize
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        scheduler.step()

    # Print progress
    print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(train_data_loader)}")

model.eval()
total_eval_accuracy = 0

for batch in validation_data_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, labels)

    logits = outputs[1]
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == labels).cpu().numpy().mean() * 100
    total_eval_accuracy += accuracy

print(f"Validation Accuracy: {total_eval_accuracy / len(validation_data_loader)}")



# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]

# class TransformerModel(nn.Module):
#     def __init__(self, ntoken, d_model, nhead, nhid, nlayers, dropout=0.5):
#         super().__init__()
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(d_model)
#         encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, nhid, dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
#         self.encoder = nn.Embedding(ntoken, d_model)
#         self.d_model = d_model
#         self.decoder = nn.Linear(d_model, ntoken)

#         self.init_weights()

#     def generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)

#     def forward(self, src, src_mask):
#         src = self.encoder(src) * math.sqrt(self.d_model)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, src_mask)
#         output = self.decoder(output)
#         return output

# # Example configuration for the model
# ntokens = 20000 # the size of vocabulary
# emsize = 200 # embedding dimension
# nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
# nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# nhead = 2 # the number of heads in the multiheadattention models
# dropout = 0.2 # the dropout value

# model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)

# # Example inputs for the model
# src = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]) # Example source sequence
# src_mask = model.generate_square_subsequent_mask(src.size(0)) # Mask for source sequence

# # Forward pass through the model
# output = model(src, src_mask)

# output.shape # Output tensor shape

