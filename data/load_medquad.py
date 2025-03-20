import os
import xml.etree.ElementTree as ET


from context import get_answer

def load_medquad(data_dir = '/home/hanoon/mini-project/medQuad/MedQuAD'):
    qa_pairs = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.xml'):
                    file_path = os.path.join(folder_path, file_name)
                    tree = ET.parse(file_path)
                    root = tree.getroot() 
                    
                    # Get context from <Focus> element
                    context_elem = root.find('Focus')
                    context = context_elem.text if context_elem is not None else ''
                    
                    # Find the QAPairs container
                    qa_container = root.find('QAPairs')
                    
                    if qa_container is not None:
                        # Find all QAPair elements
                        for qa_pair in qa_container.findall('QAPair'):
                            question_elem = qa_pair.find('Question')
                            answer_elem = qa_pair.find('Answer')
                            if question_elem is not None and answer_elem is not None:
                                answer_text = answer_elem.text or ''
                                safe_context = context or ''
                                qa_pairs.append({
                                    'context': safe_context,
                                    'question': question_elem.text,
                                    'answer': answer_text,
                                    'answer_start': safe_context.find(answer_text) if answer_text else -1
                                })
    return qa_pairs

qa_dataset = load_medquad()
print('Total Pairs: ',len(qa_dataset))
print(qa_dataset[0])