# Optimizing the chucking process
import os
import pdfplumber
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
    
    
    """
    with pdfplumber.open("PDF_docs/doc_0.pdf") as pdf:
        text = ""
        for page in pdf.pages:
        # Begrenzung auf einen Bereich (Bounding Box) für den Haupttext
            body_box = (50, 100, page.width - 50, page.height - 100)  
            body_text = page.within_bbox(body_box).extract_text()
            if body_text:
                text += body_text + "\n"

                return text

print(get_text(file_path="PDF_docs/doc_0.pdf"))



