from dotenv import load_dotenv
load_dotenv()

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)

query = "What did the author work on before college?"
query2 = "What influenced the author's early interest in programming?"
query3 = "Which programming languages does the author recommend today?"

query_engine = index.as_query_engine(similarity_top_k=6)

nodes = query_engine.retrieve(query3)

print("\n--- RETRIEVED NODES ---")
for i, n in enumerate(nodes):
    text = n.node.text
    print(f"\nNode {i+1} ({len(text)} chars):")
    print(text[:300].replace("\n", " "))


response = query_engine.query(query3)
print("\n--- FINAL ANSWER ---")
print(response)
