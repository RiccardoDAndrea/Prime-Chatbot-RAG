# Optimizing the chucking process
import os
import pdfplumber
import re
"""
First we try to create a Research Prime
to split up the fuction. And the we can try to combine them.
So this Scirpt is a deep researchish Prime-Chatbot

Data processing
-----------------
Step 1: Extract the text

Step 2: Manipualte the document so that we get the Meta data 
        (Author, Title, Summary)

Step3:  If we extract text we only want the body of the text
        not the title what is written on every page page number and 
        citztion

Step 4
"""

def get_text(file_path: str) -> str:
    """
    Extract the text from the add pdf file
    
    Args:
        file_path (str): File path to the PDF-Document.
    
    Returns:
        str: Extract text as string.
    """
    text_list = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_list.append(text)
                text = "\n".join(text_list)
    return text


def extract_citation(text: str) -> tuple[str, str]:
    """
    Split a study text into body text and references section.

    Args:
        text (str): A string containing the full study text.
    
    Returns:
        tuple[str, str]: 
            - body (str): The study body text.
            - reference (str): The extracted reference section.
    """
    # Case-insensitive split at "References"
    parts = re.split(r"\breferences\b", text, flags=re.IGNORECASE, maxsplit=1)
    
    body = parts[0].strip()  # Everything before "References"
    reference = parts[1].strip() if len(parts) > 1 else ""  # Everything after "References"

    return body, reference


def inital_reader(file_path=str):
    text = get_text(file_path = file_path)
    body, reference = extract_citation(text = text)

    return body, reference


body, reference = inital_reader(file_path="PDF_docs/doc_3.pdf")
print(body)

# text = get_text("PDF_docs/doc_0.pdf")
# body, reference  = extract_citazion(text = text)


