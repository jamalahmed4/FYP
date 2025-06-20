import os 

os.system('pip install -q tqdm')
os.system('pip install -q pikepdf')
os.system('pip install -q pdf2image')
os.system('pip install -q pytesseract')
os.system('pip install -q pillow_heif')
os.system('pip install -q pdfminer.six')
os.system('pip install -q unstructured')
os.system('pip install -q opencv-python')
os.system('pip install -q unstructured_inference')
os.system('pip install -q unstructured_pytesseract')

os.system('sudo apt-get update')
os.system('sudo apt-get install -y poppler-utils')
os.system('sudo apt-get install -y tesseract-ocr')
os.system('sudo apt-get install -y libgl1-mesa-glx')

from unstructured.partition.pdf import partition_pdf
from tqdm import tqdm

def extract_text_from_pdf(pdf_paths) : 

    text = ''

    for path in tqdm(
        pdf_paths , 
        total = len(pdf_paths) , 
        desc='Extracting text from PDFs' , 
    ) : 

        elements = partition_pdf(path)
        text += ' '.join([
            element.text 
            for element 
            in elements
        ])

    return text