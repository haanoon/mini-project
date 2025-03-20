from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load pre-trained Bio_ClinicalBERT
model_name = "emilyalsentzer/Bio_ClinicalBERT"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Sample Question and Context
question = "What are the symptoms of diabetes?"
context = "Diabetes is a chronic disease where the body struggles to regulate blood sugar levels due to insufficient insulin production or the body's inability to use insulin effectively, leading to high blood glucose levels (hyperglycemia) that can damage vital organs like the eyes, kidneys, nerves, and heart over time if left unmanaged; this occurs when the pancreas, which produces insulin, either doesn't create enough of the hormone or the body's cells become resistant to its effects, preventing glucose from entering cells for energy usage. "

# Get the answer
result = qa_pipeline(question=question, context=context)
print(result)
