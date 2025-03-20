from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json

# 1. Create Dataset Class for Preprocessed Data
class QADataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'token_type_ids': torch.tensor(self.encodings['token_type_ids'][idx]),
            'start_positions': torch.tensor(self.encodings['start_positions'][idx]),
            'end_positions': torch.tensor(self.encodings['end_positions'][idx])
        }
    
    def __len__(self):
        return len(self.encodings['input_ids'])

# 2. Load Your Preprocessed Data
def load_processed_data(path):
    data = torch.load(path)
    return QADataset(data)

# 3. Initialize Datasets
train_dataset = load_processed_data("train_processed.pt")
val_dataset = load_processed_data("test_processed.pt")  # Or your validation file

# 4. Load Model and Tokenizer
model_name = "medicalai/ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 5. Training Arguments (Keep your existing setup)
training_args = TrainingArguments(
    output_dir="./clinicalbert-qa-results",
    evaluation_strategy="steps",
    eval_steps=200,
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# 7. Start Training
trainer.train()

# 8. Save Model
trainer.save_model("./clinicalbert-qa-final")
