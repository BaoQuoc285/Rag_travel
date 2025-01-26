from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank


def load_compression_retriever(db):
    # Khởi tạo retriever từ database
    retriever = db.as_retriever()

    # Tạo RankLLMRerank compressor
    compressor = FlashrankRerank(top_n =6)

    # Tạo ContextualCompressionRetriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    return compression_retriever

def query_vector_database(compression_retriever, query, k=5):
    # Truy vấn tài liệu
    docs = compression_retriever.invoke(query)
    return docs