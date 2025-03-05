from datasets import load_dataset
import json
from transformers import AutoTokenizer
def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    # Tokenize with both question and context
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    # Handle answer positions
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer_list = examples["answers"][sample_idx]  # This is a list of answer dicts
        
        if answer_list:  # Check if there are answers
            first_answer = answer_list[0]  # Take the first answer
            start_char = first_answer["answer_start"]
            end_char = start_char + len(first_answer["text"])
            
            # Find token positions
            seq_ids = inputs.sequence_ids(i)
            token_start_index = 0
            while seq_ids[token_start_index] != 1:  # Find context start
                token_start_index += 1
                
            token_end_index = len(offsets) - 1
            while seq_ids[token_end_index] != 1:  # Find context end
                token_end_index -= 1
                
            # Check if answer is within the context
            if offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)
            else:
                start_positions.append(0)
                end_positions.append(0)
        else:
            start_positions.append(0)
            end_positions.append(0)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    
    return inputs
    
    
    


def flatten_squad(examples):
    # Initialize lists with empty values for every original example
    ids = []
    questions = []
    contexts = []
    answers = []
    
    for doc in examples:
        if isinstance(doc, str):
            try:
                doc = json.loads(doc)
            except json.JSONDecodeError:
                continue
                
        for para in doc.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                ids.append(qa.get("id", ""))
                questions.append(qa.get("question", ""))
                contexts.append(context)
                answers.append(qa.get("answers", []))
    
    # Ensure we maintain 1:1 mapping with input examples
    return {
        "new_id": ids or [""]*len(examples),
        "new_question": questions or [""]*len(examples),
        "new_context": contexts or [""]*len(examples),
        "new_answers": answers or [[]]*len(examples)
    }

def preprocess():
    # Load dataset with correct data field structure
    raw_dataset = load_dataset('json', 
        data_files={'train':'train.json', 'val': 'test.json'},
        field='data'  # Load the main data array
    )

    # Process the nested structure
    def extract_qas(example):
        return {
            "question": example["qas"]["question"],
            "id": example["qas"]["id"],
            "context": example["context"],
            "answers": example["qas"]["answers"]
        }

    processed = raw_dataset.map(
        lambda x: {
            # Add nested loop to handle batch→examples→paragraphs
            "question": [
                qa["question"] 
                for example_paragraphs in x["paragraphs"]  # Each example's paragraphs
                for para in example_paragraphs            # Each paragraph in example
                for qa in para["qas"]                      # Each QAS in paragraph
            ],
            "context": [
                para["context"]
                for example_paragraphs in x["paragraphs"]
                for para in example_paragraphs
                for qa in para["qas"]
            ],
            "answers": [
                qa["answers"]
                for example_paragraphs in x["paragraphs"]
                for para in example_paragraphs
                for qa in para["qas"]
            ],
            "id": [
                qa["id"]
                for example_paragraphs in x["paragraphs"]
                for para in example_paragraphs
                for qa in para["qas"]
            ]
        },
        batched=True,
        remove_columns=raw_dataset["train"].column_names
    )

    # Then apply tokenization
    tokenized = processed.map(
        preprocess_function,
        batched=True,
        remove_columns=["id", "question", "context", "answers"]
    )

    return tokenized
# preprocess()