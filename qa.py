import os
import sys
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def extract_text(filename):
    reader = PdfReader(filename)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text, size=1500):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start = start + size
    return chunks


def main():
    filename = sys.argv[1]
    text = extract_text(filename)
    chunks = chunk_text(text)
    print(len(chunks))


main()