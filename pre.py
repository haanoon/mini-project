from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import json

def preprocess_function(examples):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    # Tokenize inputs with truncation and padding
    inputs = tokenizer(
        examples["question"],
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Map answer positions to token space
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = examples["answers"][sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        
        # Find token positions
        token_start_index = 0
        while offset[token_start_index][0] < start_char:
            token_start_index += 1
            
        token_end_index = len(offset) - 1
        while offset[token_end_index][1] > end_char:
            token_end_index -= 1
            
        start_positions.append(token_start_index)
        end_positions.append(token_end_index)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def flatten_squad(examples):
    new_examples = []
    
    # Debug the structure
    print("Example structure:", examples.keys())
    print("First example:", examples["data"][:1])
    
    # Ensure data is properly parsed as JSON
    for doc in examples["data"]:
        if isinstance(doc, str):
            # If doc is a string, it needs to be parsed
            doc = json.loads(doc)
        
        for para in doc["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                new_examples.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answers": qa["answers"]
                })
    return {"examples": new_examples}

def preprocess():
    # Load with correct field specification
    raw_dataset = load_dataset('json', 
        data_files={
            'train': 'medquad_train.json',
            'val': 'medquad_val.json'
        },
        field='data'  # Critical addition to access the SQuAD data array
    )
    
    # Flatten structure
    flattened = raw_dataset.map(
        flatten_squad,
        batched=True,
        remove_columns=raw_dataset["train"].column_names
    )
    
    # Now preprocess the flattened data
    tokenized_dataset = flattened.map(
        preprocess_function,
        batched=True,
        remove_columns=flattened["train"].column_names
    )
    
    return tokenized_dataset
    
