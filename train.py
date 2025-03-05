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
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# # 5. Training Arguments (Keep your existing setup)
# training_args = TrainingArguments(
#     output_dir="./clinicalbert-qa-results",
#     evaluation_strategy="epoch",
#     learning_rate=3e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     fp16=True,
#     remove_unused_columns=False
# )

# # 6. Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer
# )

# # 7. Start Training
# trainer.train()

# # 8. Save Model
# trainer.save_model("./clinicalbert-qa-final")

training_args = TrainingArguments(
    output_dir="./clinicalbert-qa-results",
    evaluation_strategy="epoch",        # Evaluate every epoch
    save_strategy="steps",             # Save checkpoints every X steps
    save_steps=500,                    # Save checkpoint every 500 steps
    save_total_limit=2,                # Keep only last 2 checkpoints
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,       # For final model selection
    metric_for_best_model="loss",      # Alternative: "eval_f1"
    greater_is_better=False,           # For loss (lower is better)
    fp16=True,
    remove_unused_columns=False,
    report_to="tensorboard",           # Track progress in TensorBoard
    logging_steps=100                  # Log metrics every 100 steps
)

# 2. Trainer Initialization (unchanged)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# 3. Start Training with Resume Capability
trainer.train(resume_from_checkpoint=True)  # Auto-resume if checkpoint exists

# 4. Save Final Model (unchanged)
trainer.save_model("./clinicalbert-qa-final")
