import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import DataFrameLoader

DATA_PATH = "/Users/deltae/Downloads/IITG Research/RAG Dataset/College Essentials/HireSift/data/csv"
FAISS_PATH = "/Users/deltae/Downloads/IITG Research/RAG Dataset/College Essentials/HireSift/vectorstore-pdf"
EMBEDDING_MODEL = "voyage-large-2-instruct"


def ingest(df: pd.DataFrame, content_column: str, embedding_model):
  loader = DataFrameLoader(df, page_content_column=content_column)

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1024,
    chunk_overlap = 500
  )

  documents = loader.load()
  document_chunks = text_splitter.split_documents(documents)

  vectorstore_db = FAISS.from_documents(document_chunks, embedding_model)
  return vectorstore_db