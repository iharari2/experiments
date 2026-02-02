from dotenv import load_dotenv
load_dotenv()

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

docs = SimpleDirectoryReader("data/raw").load_data()

print("\n--- INGESTED DOCUMENTS ---")
for d in docs:
    print(f"Doc length: {len(d.text)} chars")
    print(d.text)
    print("-------------------------")

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
nodes = query_engine.retrieve("What is LlamaIndex?")
for i, node in enumerate(nodes):
    print(f"\nChunk {i+1}:")
    print(node.node.text)

response = query_engine.query("What's the latest LlamaIndex version?")
print("\n--- FINAL ANSWER ---")
print(response)


