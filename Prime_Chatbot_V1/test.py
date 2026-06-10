import os

# Absoluter Pfad zum Ordner, in dem Prime-Chatbot.py liegt
current_dir = os.path.dirname(os.path.abspath(__file__))

# Erstelle die Pfade dynamisch
example_pdfs = {
    "The economic potential of generative AI": os.path.join(current_dir, "PDF_docs", "the-economic-potential-of-generative-ai-the-next-productivity-frontier-vf.pdf"),
    "Overcoming huge challenges in cancer": os.path.join(current_dir, "PDF_docs", "WIREs Mechanisms of Disease - 2013 - Roukos - Genome network medicine  innovation to overcome huge challenges in cancer.pdf"),
}

# Kleiner Debug-Test für dich:
for name, path in example_pdfs.items():
    if not os.path.exists(path):
        print(f"❌ FEHLER: Datei für '{name}' nicht gefunden unter: {path}")
    else:
        print(f"✅ GEFUNDEN: '{name}' ist bereit.")