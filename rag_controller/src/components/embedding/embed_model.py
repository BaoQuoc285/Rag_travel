from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    def __init__(self, model_name, type_, model_kwargs=None, encode_kwargs=None):
        """
        Initialize the embedding model.

        Args:
            model_name (str): Name of the model (e.g., 'API' or HuggingFace model name).
            type_ (str): Type of task ('retrieval' or other types such as 'semantic_similarity').
            model_kwargs (dict, optional): Arguments for the model configuration (e.g., device).
            encode_kwargs (dict, optional): Arguments for encoding configurations (e.g., batch size, normalization).
        """
        self.model_name = model_name
        self.type_ = type_
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}

    def get_embedding(self):
        """
        Create and return the appropriate embedding model based on the provided configuration.

        Returns:
            embeddings: An instance of the embedding model.
        """
        if self.type_ == 'retrieval':
            if self.model_name == 'API':
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-002", 
                    task_type="retrieval_query"
                )
            else:
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs=self.model_kwargs,
                    encode_kwargs=self.encode_kwargs
                )
        else:
            if self.model_name == 'API':
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-002", 
                    task_type="semantic_similarity"
                )
            else:
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs=self.model_kwargs,
                    encode_kwargs=self.encode_kwargs
                )

        return embeddings
