import os
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from pinecone import Pinecone

load_dotenv()

required_variables = [
    "NVIDIA_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME"

]

missing_variables = [
    name for name in required_variables
    if not os.getenv(name)
]

if missing_variables:
    raise RuntimeError(f"Missing environment variables: {'.'.join(missing_variables)}")



embeddings = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")

test_vector = embeddings.embed_query("RAG Project Setup verification")

print("Embedding Dimensions:", len(test_vector))

# Pinecone Compatibility

pinecone_client = Pinecone(api_key = os.environ["PINECONE_API_KEY"])

index_name = os.environ["PINECONE_INDEX_NAME"]
index_info = pinecone_client.describe_index(name= index_name)

embedding_dimension = len(test_vector)
index_dimension = index_info.dimension

print("Embedding Dimension: ", embedding_dimension)
print("Pinecone dimension:", index_dimension)


