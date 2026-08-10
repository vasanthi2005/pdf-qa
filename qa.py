import os
import sys
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import numpy as np

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def retrieve(question,chunks,vectors,top_n=5):
    q_vector=embed(question)
    scores=[]
    for i in range(len(vectors)):
        score=similarity(q_vector,vectors[i])
        scores.append((score,i))

    scores.sort(reverse=True)
    return [(chunks[i],score) for score,i in scores[:top_n]]

def answer(question,retrieved):
    context="\n\n".join([chunk for chunk,score in retrieved])
    response=client.models.generate_content(
        model="gemini-flash-latest",
        contents=f"Answer the question using only the context below. If the context doesn't contain the answer, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    return response.text
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
    filename = sys.argv[1]
    text = extract_text(filename)
    chunks = chunk_text(text)
    
    vectors = []
    for chunk in chunks:
        vectors.append(embed(chunk))

    question="What accuracy did the prediction model achieve"
    result=retrieve(question,chunks,vectors)

    for chunk,score in result:
        print(round(score,3))
        print(chunk[:200])
        print("-----")

    print(answer(question,result))


    # a = embed("the cat sat on the mat")
    # b = embed("a feline rested on the rug")
    # c = embed("quarterly revenue increased by twelve percent")

    # print(similarity(a, b))
    # print(similarity(a, c))


main()
