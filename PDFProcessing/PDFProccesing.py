import os
import pdfplumber
import re
from pypdf import PdfReader

"""
First we try to create a Research Prime
to split up the fuction. And the we can try to combine them.
So this Scirpt is a deep researchish Prime-Chatbot

Problem: Every PDF is written diffrently

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

# python PDFProcessing/PDFProccesing.py > outputs/refoutput.txt


class ReaderPDF:
    def __init__(self, file_path):
        self.file_path = file_path


    def get_text(self) -> str:
        """
        Extract the text from the given PDF file.
        
        Returns:
            str: Extracted text as a string.
        """
        text_list = []

        with PdfReader(self.file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_list.append(text)
                    text = text.join(text_list)
        
        return text


    def extract_citation(self, text: str) -> tuple[str, str]:
        """
        Split a study text into body text and references section.

        Args:
            text (str): A string containing the full study text.
        
        Returns:
            tuple[str, str]: 
                - body (str): The study body text.
                - reference (str): The extracted reference section.
        """
        parts = re.split(r"\breferences\b", text, flags=re.IGNORECASE, maxsplit=1)
        
        body = parts[0].strip()  # Everything before "References"
        reference = parts[1].strip() if len(parts) > 1 else ""  # Everything after "References"

        return body, reference


    def get_metadata(self) -> str:
        """
        Retrieve the title from the meta data from the PDF Document.

        Returns:
            str:
                - title of the Document
        """
        pdf = pdfplumber.open(self.file_path)
        metadata = pdf.metadata  # Extrahiere Metadaten
        if metadata and "Title" in metadata and metadata["Title"]:
            title = metadata['Title']
            return title
        else:
            title = "No Title"
            return title


    def initial_reader(self):
        text = self.get_text()
        body, reference = self.extract_citation(text)
        title = self.get_metadata()
        return title, body, reference
    


# Create an instance of ReaderPDF
reader = ReaderPDF(file_path="PDF_docs/doc_1.pdf")

# Extract tile, text and citations
title, body, reference = reader.initial_reader()

print(title)