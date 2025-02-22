from datasets import load_dataset

dataset = load_dataset("medmcqa")
print(dataset["train"][0])  # View first question
