from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

def read_pdf(file):
    pdf_reader = PyPDFLoader(file)

    data = pdf_reader.load()
    
    return data

def chunk_list(content_list,chunk_size,chunk_overlap):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators =["\n\n", "\n", " ", ""])
    
    recursive_chunks = text_splitter.split_documents(content_list)
    
    return recursive_chunks