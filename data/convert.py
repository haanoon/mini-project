import json
from collections import defaultdict
from sklearn.model_selection import train_test_split

def validate_answer_span(context, answer_text, answer_start):
    """Ensure answer exactly matches context substring"""
    if answer_start == -1:
        return False
    end_pos = answer_start + len(answer_text)
    return context[answer_start:end_pos] == answer_text

def convert_to_squad(qa_pairs, output_file="medquad_squad.json"):
    squad_data = {
        "version": "v2.0",
        "data": []
    }

    # Group QA pairs by context
    context_map = defaultdict(list)
    for idx, pair in enumerate(qa_pairs):
        context_map[pair['context']].append({
            "id": str(idx),
            "question": pair['question'],
            "answer": pair['answer'],
            "answer_start": pair['answer_start']
        })

    # Build SQuAD structure
    for context_idx, (context, qas) in enumerate(context_map.items()):
        squad_entry = {
            "title": f"MedicalDoc_{context_idx}",
            "paragraphs": []
        }

        paragraph = {
            "context": context,
            "qas": []
        }

        valid_qa_count = 0
        for qa in qas:
            # Validate answer span
            is_valid = validate_answer_span(context, qa['answer'], qa['answer_start'])
            
            qa_entry = {
                "id": f"doc{context_idx}_qa{valid_qa_count}",
                "question": qa['question'],
                "answers": [],
                "is_impossible": not is_valid
            }

            if is_valid:
                qa_entry["answers"].append({
                    "text": qa['answer'],
                    "answer_start": qa['answer_start']
                })
            else:
                # Handle unanswerable questions
                qa_entry["plausible_answers"] = []

            paragraph['qas'].append(qa_entry)
            valid_qa_count += 1

        squad_entry['paragraphs'].append(paragraph)
        squad_data['data'].append(squad_entry)

    # Save with validation
    with open(output_file, 'w') as f:
        json.dump(squad_data, f, indent=2)


    # Split contexts (80% train, 20% validation)
    train_contexts, val_contexts = train_test_split(
        squad_data['data'], 
        test_size=0.2, 
        random_state=42
    )

    # Save separate files
    with open('medquad_train.json', 'w') as f:
        json.dump({"version": squad_data['version'],
                "data": train_contexts
                }, f)
        
    with open('medquad_val.json', 'w') as f:
        json.dump({"version":squad_data['version'],
                   "data": val_contexts
                   }, f)

    print(f"Converted {len(qa_pairs)} pairs to SQuAD format with {len(context_map)} contexts")
    return squad_data

