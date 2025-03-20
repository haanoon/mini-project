import json
import os

def convert_to_squad(output_file, input_dir='./med'):
    """
    Converts medical JSON files to SQuAD format
    
    Args:
        output_file (str): Path for output SQuAD-formatted file
        input_dir (str): Directory containing subfolders with .json files
    """
    squad_data = {"data": []}
    
    # Process each SUBFOLDER in input directory
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            
            # Process each FILE in the subfolder
            for filename in os.listdir(folder_path):
                if filename.endswith(".json"):
                    with open(os.path.join(folder_path, filename)) as f:
                        data = json.load(f)
                        
                        # Create SQuAD entry
                        squad_entry = {
                            "title": data["title"],
                            "paragraphs": []
                        }
                        
                        # Convert Q&A pairs
                        for qa in data["qas"]:
                            # Create a paragraph for EACH ANSWER
                            for answer in qa["answers"]:
                                paragraph = {
                                    "context": answer["text"],  # Use answer as context
                                    "qas": [{
                                        "id": qa["id"],
                                        "question": qa["question"],
                                        "answers": [{
                                            "text": answer["text"],
                                            "answer_start": 0  # Answer starts at beginning of context
                                        }]
                                    }]
                                }
                                squad_entry["paragraphs"].append(paragraph)
                        
                        squad_data["data"].append(squad_entry)
    
    # Moved outside the loops to save ONCE after processing all files
    with open(output_file, "w") as f:
        json.dump(squad_data, f, indent=2)

# Usage example
convert_to_squad(output_file= "squad_formatted_data.json")