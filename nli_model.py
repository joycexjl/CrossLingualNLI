import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from alignment_model import FastTextAlignmentModel

class CrossLingualNLIModel(nn.Module):
    def __init__(self, source_lang, target_lang, num_labels=3):
        super(CrossLingualNLIModel, self).__init__()

        # Initialize the alignment model
        self.alignment_model = FastTextAlignmentModel(source_lang, target_lang)

        # Initialize the BERT model for sequence classification
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)

        # Additional layer to integrate alignment information
        bert_hidden_size = 768
        self.alignment_integration_layer = nn.Linear(bert_hidden_size, bert_hidden_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(bert_hidden_size)

    def preprocess_data(self, sentence_source, sentence_target):
        # Get alignments
        alignments = self.alignment_model.align_sentences(sentence_source, sentence_target)

        # Process alignments
        source_tokens = sentence_source.split()  # Simple tokenization, replace with appropriate method
        target_tokens = sentence_target.split()  # Simple tokenization, replace with appropriate method

        for src_idx, tgt_idx in alignments:
            if src_idx < len(source_tokens) and tgt_idx < len(target_tokens):
                source_tokens[src_idx] += ' <ALIGN>'
                target_tokens[tgt_idx] += ' <ALIGN>'

        # Rejoin tokens into a string
        aligned_source_sentence = ' '.join(source_tokens)
        aligned_target_sentence = ' '.join(target_tokens)

        # Tokenize the aligned sentences
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        encoded_input = tokenizer.encode_plus(
            aligned_source_sentence, 
            aligned_target_sentence, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True, 
            padding='max_length'
        )

        return encoded_input


    def forward(self, input_ids, attention_mask):
        # Forward pass through the BERT model
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden states
        last_hidden_states = outputs[0]

        # Process the last hidden states with the alignment integration layer
        alignment_enhanced_features = self.alignment_integration_layer(last_hidden_states)
        alignment_enhanced_features = self.dropout(alignment_enhanced_features)
        alignment_enhanced_features = self.layer_norm(alignment_enhanced_features + last_hidden_states)

        return outputs.logits

# class BertNLIModel(nn.Module):
#     def __init__(self, num_labels=3):  # 3 labels for NLI: entailment, neutral, contradiction
#         super(BertNLIModel, self).__init__()
#         self.bert = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)

#     def forward(self, input_ids, attention_mask, labels=None):
#         output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         return output

# num_epochs = 3  # Define the number of epochs

# # Instantiate the model
# model = BertNLIModel()

# # Define the optimizer
# optimizer = AdamW(model.parameters(), lr=5e-5)

# # Scheduler for learning rate decay
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_data_loader) * num_epochs)

# # Define the loss function
# criterion = nn.CrossEntropyLoss()

# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0

#     for batch in train_data_loader:
#         # Move batch to device
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)

#         # Clear previous gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(input_ids, attention_mask, labels)
#         loss = outputs[0]

#         # Backward pass and optimize
#         loss.backward()
#         total_loss += loss.item()
#         optimizer.step()
#         scheduler.step()

#     # Print progress
#     print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(train_data_loader)}")

# model.eval()
# total_eval_accuracy = 0

# for batch in validation_data_loader:
#     input_ids = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     labels = batch['labels'].to(device)

#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask, labels)

#     logits = outputs[1]
#     predictions = torch.argmax(logits, dim=-1)
#     accuracy = (predictions == labels).cpu().numpy().mean() * 100
#     total_eval_accuracy += accuracy

# print(f"Validation Accuracy: {total_eval_accuracy / len(validation_data_loader)}")



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

