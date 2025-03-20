API_KEY = 'AIzaSyCtD44gm_4qT7vBZ8T0B8jVcsCaA0rBrxA'

import google.generativeai as genai
import re

def preprocess_text(text):

    # Remove headings (lines starting with #, **, or similar markdown-like patterns)
    text = text.replace("\n", " ")
    text = re.sub(r'^[#*]+ .*', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_context(question):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash') #or gemini-pro-vision, gemini-flash
    response = model.generate_content(f"Provide a detailed medical explanation for: {question} as a paragraph in detail")

    return preprocess_text(response.text)


def get_answer(question):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash') #or gemini-pro-vision, gemini-flash
    response = model.generate_content(f"Provide a detailed answer for: {question} as a paragraph in detail")

    return preprocess_text(response.text)

print(get_context('what is diabetes? What are the glucose level of normal people'))