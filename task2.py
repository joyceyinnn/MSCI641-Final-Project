import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, AdamW

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available")

# Load the pre-trained DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

# Define a custom dataset class for spoilers
class SpoilerDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        inputs = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        inputs['labels'] = inputs.input_ids.detach().clone()

        # Randomly mask some tokens for training the model to generate spoilers
        mask_indices = torch.bernoulli(torch.full(inputs.input_ids.shape, 0.15)).bool()
        inputs.input_ids[mask_indices] = self.tokenizer.mask_token_id

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': inputs['labels'].flatten()
        }

# Load the training data 
train_path = r'C:\Users\User\Desktop\msci641\train.jsonl'
train_data = pd.read_json(train_path, lines=True)

# Combine 'postText' and 'targetParagraphs' into a single input text sequence
train_data['input_text'] = train_data.apply(lambda row: ' '.join(row['postText']) + ' ' + ' '.join(row['targetParagraphs']), axis=1)

# Create a DataLoader for batching and shuffling
train_texts = train_data['input_text'].tolist()
train_dataset = SpoilerDataset(train_texts, tokenizer, max_len=512)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# # Set up the AdamW optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3

print('Start training')
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 保存微调后的模型
model.save_pretrained('fine_tuned_distil_mlm')
tokenizer.save_pretrained('fine_tuned_distil_tokenizer')

# Save the fine-tuned model and tokenizer
model = DistilBertForMaskedLM.from_pretrained('fine_tuned_distil_mlm')
tokenizer = DistilBertTokenizer.from_pretrained('fine_tuned_distil_tokenizer')

# Generate function of spoilers
def safe_predict(text, max_length=20):
    encoded_input = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
    encoded_input = torch.tensor(encoded_input).unsqueeze(0)  # 确保是二维张量 (1, seq_length)

    with torch.no_grad():
        outputs = model(encoded_input)

    predicted_ids = outputs.logits.argmax(dim=-1).squeeze().tolist()
    predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)

    return predicted_text

# Load the test data
test_path = r'C:\Users\User\Desktop\msci641\test.jsonl'
test_data = pd.read_json(test_path, lines=True)

# Combine 'postText' and 'targetParagraphs' into a single input text sequence for the test data
test_data['input_text'] = test_data.apply(lambda row: ' '.join(row['postText']) + ' ' + ' '.join(row['targetParagraphs']), axis=1)

# Generate spoilers for the test data
test_data['spoiler'] = test_data['input_text'].apply(lambda x: safe_predict(x, max_length=20))

# Ensure there is an 'id' column
test_data['id'] = test_data.index

# Save the generated spoilers to CSV file
test_data[['id', 'spoiler']].to_csv('generated_spoilers.csv', index=False)
print("Spoilers saved to 'generated_spoilers.csv'")
