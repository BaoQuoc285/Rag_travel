import os
from src.components.vector_store.chunk import read_pdf, chunk_list
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from src.components.embedding.embed_model import EmbeddingModel
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
def load_sparse_embeddings(model_name="Qdrant/bm25"):
       
    sparse_embeddings = FastEmbedSparse(model_name=model_name)
    return sparse_embeddings
    

def load_dense_embeddings( model_name, device, batch_size, normalize_embeddings=False):

    encode_kwargs = {
        "batch_size": batch_size,
        "normalize_embeddings": normalize_embeddings
    }

    model_kwargs = {"device": device}

    model = EmbeddingModel(
        model_name=model_name,
        type_="indexing",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    embeddings = model.get_embedding()

    return embeddings

def load_and_chunk_pdf( file_path, chunk_size=1024, chunk_overlap=100):
    dataset = read_pdf(file_path)
    return chunk_list(content_list=dataset, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class QdrantRAG:
    def __init__(self, api_key_env="QDRANT_API_KEY", url_env="QDRANT_URL"):
        
        self.qdrant_api_key =api_key_env
        self.qdrant_url = url_env
        
        
        if not self.qdrant_api_key or not self.qdrant_url:
            raise ValueError("Qdrant API key or URL not found in environment variables.")

        self.sparse_embeddings = load_sparse_embeddings()
        self.embeddings = load_dense_embeddings(
            model_name="BAAI/bge-m3",
            device="cpu",
            batch_size=100)

    def load_data_into_qdrant(self, file_data ,collection_name="my_documents"):

        
        if not self.embeddings:
            raise ValueError("Embeddings are not loaded. Please load embeddings first.")
        if not self.sparse_embeddings:
            raise ValueError("Sparse embeddings are not loaded. Please load sparse embeddings first.")
        #load data 
        txt = load_and_chunk_pdf(file_data)

        print("Start loading data into Qdrant...")
        qdrant = QdrantVectorStore.from_documents(
            txt,
            embedding=self.embeddings,
            url=self.qdrant_url,
            prefer_grpc=True,
            api_key=self.qdrant_api_key,
            sparse_embedding=self.sparse_embeddings,
            collection_name=collection_name,
            retrieval_mode=RetrievalMode.HYBRID
        )
        print("Finished loading data into Qdrant.")
        return qdrant
    def load_vector_database(self,collection_name="my_documents"):
        qdrant_client = QdrantClient(
            url= self.qdrant_url,
            api_key= self.qdrant_api_key
        )

        db = QdrantVectorStore(client=qdrant_client, embedding=self.embeddings, collection_name=collection_name)
        return db
