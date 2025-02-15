# Optimizing the chucking process
import os
import pypdf
# creating a pdf reader object
reader = pypdf.PdfReader('PDF_docs/doc.pdf')

# print the number of pages in pdf file
print(reader.pages[0].extract_text())
#print(len(reader.pages))
