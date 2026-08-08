import os
import sys
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import numpy as np

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


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


def embed(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def main():
    # filename = sys.argv[1]
    # text = extract_text(filename)
    # chunks = chunk_text(text)
    #
    # vectors = []
    # for chunk in chunks:
    #     vectors.append(embed(chunk))

    a = embed("the cat sat on the mat")
    b = embed("a feline rested on the rug")
    c = embed("quarterly revenue increased by twelve percent")

    print(similarity(a, b))
    print(similarity(a, c))


main()
