from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load pre-trained Bio_ClinicalBERT
model_name = "emilyalsentzer/Bio_ClinicalBERT"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Sample Question and Context
question = "What are the symptoms of diabetes?"
context = "Diabetes symptoms include frequent urination, excessive thirst, and increased hunger."

# Get the answer
result = qa_pipeline(question=question, context=context)
print(result)
