from dotenv import load_dotenv
load_dotenv()

from llama_index.core import SimpleDirectoryReader
from llama_index.core.text_splitter import SentenceSplitter

docs = SimpleDirectoryReader("data/").load_data()

splitter = SentenceSplitter(chunk_size=128, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(docs)

print(f"Docs: {len(docs)}")
print(f"Nodes: {len(nodes)}")

for i, n in enumerate(nodes[:10]):  # first 10
    t = n.text
    print(f"Node {i+1}: {len(t)} chars")
    print(t[:300].replace('\n', ' '))
    print("-" * 40)

