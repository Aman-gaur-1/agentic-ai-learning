import os

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == "__main__":

    print('Ingesting........')
    loader = UnstructuredLoader('consoleflareblog.txt',chunking_strategy='basic',max_characters=10000)
    document = loader.load()

    # Creating Chunk

    print('Splitting.....')

    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    text = text_splitter.split_documents(document)
    print(f'Created {len(text)} Chunks')
    # print(text)

    # for i, chunk in enumerate(text):
    #     print(f'\n --- Chunk {i + 1} ---')
    #     print(chunk.page_content[:300])


    # Ingesting
    embeddings = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")

    print('Ingesting.....')
    PineconeVectorStore.from_documents(text,embeddings,index_name=os.environ["PINECONE_INDEX_NAME"])
    print('FINISH'.center(50,'-'))
