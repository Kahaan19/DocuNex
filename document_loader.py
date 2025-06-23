import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.schema import Document

def load_documents_from_files(file_paths: List[str]) -> List[Document]:
    """
    Loads text content from a list of files (PDF, DOCX, TXT) and returns LangChain Document objects.
    """
    docs = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        
        ext = os.path.splitext(path)[-1].lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext == ".docx":
                loader = Docx2txtLoader(path)
            elif ext == ".txt":
                loader = TextLoader(path)
            else:
                print(f"Unsupported file type: {ext}")
                continue

            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    return docs
