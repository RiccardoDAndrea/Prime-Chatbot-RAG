# Optimizing the chucking process
import os
import pypdf
# creating a pdf reader object
reader = pypdf.PdfReader('PDF_docs/doc.pdf')

# print the number of pages in pdf file
#print(len(reader.pages[0].extract_text()))



page = reader.pages[0]

parts = []


def visitor_body(text, cm, tm, fontDict, fontSize):
    y = tm[5]
    if y > 100 and y < 250:
        parts.append(text)


page.extract_text(visitor_text=visitor_body)
text_body = "".join(parts)

print(text_body)

# Function returns the pages
#print(len(reader.pages))
