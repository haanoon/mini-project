from sklearn.model_selection import train_test_split
import json

def split_data():
    with open('out.json', 'r') as f:
        data = json.load(f)

    train, test = train_test_split(data, test_size=0.2, random_state=42)

    with open('medquad_squad_train.json', 'w') as f:
        json.dump(train, f)

    with open('medquad_squad_test.json', 'w') as f:
        json.dump(test, f)

split_data()