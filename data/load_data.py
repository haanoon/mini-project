from load_medquad import load_medquad
from convert import convert_to_squad
import json
from split import split_data
# Load and convert with validation
qa_dataset = load_medquad()
squad_data = convert_to_squad(qa_dataset)

# Optional: Verify first entry
print("Sample SQuAD entry:")
print(json.dumps(squad_data['data'][0]['paragraphs'][0]['qas'][0], indent=2))
