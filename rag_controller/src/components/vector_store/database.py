import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from src.components.embedding.embed_model import EmbeddingModel


import os
from dotenv import load_dotenv

load_dotenv()

qdrant_api = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
print(f"Qdrant API Key: {qdrant_api}")
print(f"Qdrant URL: {qdrant_url}")





sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
encode_kwargs = {
    "batch_size": 100,  # Kích thước batch khi mã hóa \n
    "normalize_embeddings": False  # Chuẩn hóa các vector nhúng (thường để vector có chuẩn bằng 1)
}
model_kwargs={'device': 'cpu'}

model_name = EmbeddingModel(
    model_name = "BAAI/bge-m3",
    type_ ="indexing",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )

embeddings = model_name.get_embedding()

print("embedding: ",embeddings)
print("Finish loading model.....")

from src.components.vector_store.chunk import  read_pdf,chunk_list

file_path = "/mnt/c/Users/PC/CS311_Travel_RAG/data/blog/dulich.pdf"
dataset = read_pdf(file_path)

txt = chunk_list(content_list = dataset, chunk_size = 1024 ,chunk_overlap= 100)

print("start to load data into qdrant ..........")
qdrant = QdrantVectorStore.from_documents(
    txt,
    embedding=embeddings,
    url=qdrant_url,
    prefer_grpc=True,
    api_key=qdrant_api,
    sparse_embedding=sparse_embeddings,
    collection_name="my_documents",
    retrieval_mode=RetrievalMode.HYBRID,
)

print("finish load data in qdrant. ")

print(qdrant)