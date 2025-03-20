from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import json

from transformers import EvalPrediction
import numpy as np

def compute_metrics(p: EvalPrediction):
    """Calculate F1 and exact match metrics"""
    start_logits, end_logits = p.predictions
    start_positions = p.label_ids[0]
    end_positions = p.label_ids[1]
    
    # Get predicted spans
    pred_starts = np.argmax(start_logits, axis=1)
    pred_ends = np.argmax(end_logits, axis=1)
    
    # Calculate exact matches
    exact_matches = np.logical_and(
        pred_starts == start_positions,
        pred_ends == end_positions
    ).mean()
    
    # Calculate F1 scores
    f1_scores = []
    for i in range(len(pred_starts)):
        pred_span = set(range(pred_starts[i], pred_ends[i]+1))
        true_span = set(range(start_positions[i], end_positions[i]+1))
        
        overlap = len(pred_span & true_span)
        precision = overlap / len(pred_span) if pred_span else 0
        recall = overlap / len(true_span) if true_span else 0
        f1 = 2 * (precision * recall) / (precision + recall + 1e-20)
        f1_scores.append(f1)
    
    return {
        "exact_match": exact_matches,
        "f1": np.mean(f1_scores)
    }



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
    eval_steps=500,
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Add metrics calculation
)

# 7. Start Training
trainer.train()

# After training completes
final_metrics = trainer.evaluate()
print(f"Final evaluation results: {final_metrics}")

# Save metrics to file
with open("evaluation_results.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

# 8. Save Model
trainer.save_model("./clinicalbert-qa-final")


from transformers import Trainer

def evaluate_model(model_path, eval_dataset):
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    trainer = Trainer(model=model)
    
    results = trainer.predict(eval_dataset)
    print("Evaluation metrics:", results.metrics)
    
    return results

# Usage
eval_results = evaluate_model(
    "./clinicalbert-qa-final", 
    val_dataset
)