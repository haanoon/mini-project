import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# Load tokenizer
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Preprocessing parameters
max_length = 384
stride = 128
pad_to_max_length = True

def validate_answer(context, answer_start, answer_text):
    """Ensure answer positions are valid and match text"""
    answer_end = answer_start + len(answer_text)
    if answer_start < 0 or answer_end > len(context):
        return False
    if context[answer_start:answer_end] != answer_text:
        return False
    return True

def preprocess_squad_data(squad_data):
    processed = {
        "input_ids": [],
        "attention_mask": [],
        "token_type_ids": [],
        "start_positions": [],
        "end_positions": []
    }

    for entry in squad_data["data"]:
        for para in entry["paragraphs"]:
            context = para["context"]
            if not context.strip():  # Skip empty contexts
                continue
                
            for qa in para["qas"]:
                answer = qa["answers"][0]
                answer_start = answer["answer_start"]
                answer_text = answer["text"]
                
                # Validate answer positions
                if not validate_answer(context, answer_start, answer_text):
                    print(f"Skipping invalid answer: {qa['id']}")
                    continue

                # Tokenize with error handling
                try:
                    inputs = tokenizer(
                        qa["question"],
                        context,
                        max_length=max_length,
                        truncation="only_second",
                        stride=stride,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding="max_length" if pad_to_max_length else False,
                    )
                except Exception as e:
                    print(f"Tokenization error for QID {qa['id']}: {e}")
                    continue

                # Process each window
                for i in range(len(inputs["input_ids"])):
                    sequence_ids = inputs.sequence_ids(i)
                    offset_mapping = inputs["offset_mapping"][i]

                    # Find context boundaries safely
                    try:
                        context_start = sequence_ids.index(1)
                        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
                    except ValueError:
                        continue

                    # Get character span of current window
                    window_start = offset_mapping[context_start][0]
                    window_end = offset_mapping[context_end][1]
                    answer_end_char = answer_start + len(answer_text)

                    # Check if answer is fully in window
                    if not (window_start <= answer_start and answer_end_char <= window_end):
                        continue

                    # Find token positions
                    token_start = token_end = None
                    for idx in range(context_start, context_end + 1):
                        if offset_mapping[idx][0] <= answer_start < offset_mapping[idx][1]:
                            token_start = idx
                        if offset_mapping[idx][0] < answer_end_char <= offset_mapping[idx][1]:
                            token_end = idx
                    
                    # Validate token positions
                    if token_start is None or token_end is None:
                        continue
                    if token_start > token_end:
                        continue
                    if token_end >= len(offset_mapping):
                        continue

                    # Add to processed data
                    processed["input_ids"].append(inputs["input_ids"][i])
                    processed["attention_mask"].append(inputs["attention_mask"][i])
                    processed["token_type_ids"].append(inputs["token_type_ids"][i])
                    processed["start_positions"].append(token_start)
                    processed["end_positions"].append(token_end)

    return processed

# Load and preprocess data
with open("train.json") as f:
    train_data = json.load(f)
with open("test.json") as f:
    test_data = json.load(f)

train_processed = preprocess_squad_data(train_data)
test_processed = preprocess_squad_data(test_data)

# Convert to tensors and save
torch.save({k: torch.tensor(v) for k, v in train_processed.items()}, "train_processed.pt")
torch.save({k: torch.tensor(v) for k, v in test_processed.items()}, "test_processed.pt")