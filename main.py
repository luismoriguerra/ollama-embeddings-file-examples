


import json
import os
import time
import ollama
import numpy as np
from numpy.linalg import norm

def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
    return paragraphs


def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False

    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)):
        return embeddings
    
    embeddings = [ollama.embeddings(model=modelname, prompt=chunk)["embedding"] for chunk in chunks]
    
    save_embeddings(filename, embeddings)
    return embeddings

def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [np.dot(needle, item)/(needle_norm * norm(item)) for item in haystack]
    
    return sorted(zip(similarity_scores, range(len(similarity_scores))), reverse=True)

def main():
    
    SYSTEM_PROMPT=""" You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    
    filename = "peter-pan.txt"
    paragraphs = parse_file(filename)
    start = time.perf_counter()
    embeddings = get_embeddings(filename, "llama3", paragraphs)
    print(time.perf_counter() - start)
    
    # prompt = "who is the story's protagonist?"
    prompt = input("Enter your question: ")
    prompt_embedding = ollama.embeddings(model="mistral", prompt=prompt)["embedding"]
    
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:10]
    
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT
            + "\n".join([paragraphs[item[1]] for item in most_similar_chunks])},
            {"role": "user", "content": prompt}
        ]
    )
    print("\n\n")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()