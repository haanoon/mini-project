from transformers import (
    AutoModelForQuestionAnswering, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset
import torch
from preprocess import preprocess
from pprint import pprint
import json
# Load the preprocessed data
dataset = preprocess()  # Should return tokenized data

pprint(dataset['train'])

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 2. Training Arguments Configuration
training_args = TrainingArguments(
    output_dir="./clinicalbert-qa-results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    remove_unused_columns=False  # Keep this as False now
)

# 3. Trainer Initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= dataset['train'],
    eval_dataset= dataset['val'],  # From your val split
    tokenizer=AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT"),
)

# 4. Start Training
trainer.train()

# 5. Save Final Model
trainer.save_model("./clinicalbert-qa-final")