# Optimizing the chucking process
import os
import pdfplumber

def get_text(file_path: str) -> str:
    """
    
    
    """
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text(
                x_tolerance=3, 
                y_tolerance=3, 
                layout=False, 
                x_density=7.25, 
                y_density=13, 
                line_dir_render="ttb"
            )
            if page_text:  # Avoid adding None values
                text += page_text + "\n"
        return text

print(get_text(file_path="PDF_docs/doc_0.pdf"))



