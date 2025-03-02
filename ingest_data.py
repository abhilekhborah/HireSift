import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import DataFrameLoader
from langchain.docstore.document import Document
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ingest_data')

DATA_PATH = os.getenv("DATA_PATH", "/Users/deltae/Downloads/IITG Research/RAG Dataset/College Essentials/HireSift/data/csv")
FAISS_PATH = os.getenv("FAISS_PATH", "/Users/deltae/Downloads/IITG Research/RAG Dataset/College Essentials/HireSift/vectorstore-pdf")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "voyage-large-2-instruct")


def ingest(df, content_column="Resume", embedding_model=None, chunk_size=1024, chunk_overlap=500):
    """
    Enhanced ingestion function with better error handling and flexibility
    
    Args:
        df: DataFrame or list of dictionaries with resume data
        content_column: Column/key name containing the resume text
        embedding_model: Model to use for embeddings
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        FAISS vectorstore with embedded documents
    """
    try:
        # Handle empty input case
        if df is None or (hasattr(df, 'empty') and df.empty) or (isinstance(df, list) and len(df) == 0):
            logger.warning("Empty data provided to ingest(). Creating empty vectorstore.")
            # Create a minimal document to initialize the vectorstore
            empty_doc = Document(page_content="Empty document placeholder", metadata={"source": "empty"})
            return FAISS.from_documents([empty_doc], embedding_model, distance_strategy=DistanceStrategy.COSINE)
            
        # Convert to DataFrame if we received a list
        if isinstance(df, list):
            df = pd.DataFrame(df)
            
        # Check if the content column exists
        if content_column not in df.columns:
            available_columns = ', '.join(df.columns)
            logger.error(f"Content column '{content_column}' not found in DataFrame. Available columns: {available_columns}")
            # Try to guess which column might contain the resume text
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            if text_columns:
                content_column = text_columns[0]
                logger.info(f"Using '{content_column}' as fallback content column")
            else:
                raise KeyError(f"Cannot find suitable text column in DataFrame.")
        
        # Ensure ID column exists
        if "ID" not in df.columns:
            logger.info("Adding ID column to DataFrame")
            df["ID"] = range(len(df))
        
        # Create document loader
        try:
            loader = DataFrameLoader(df, page_content_column=content_column)
            documents = loader.load()
        except Exception as e:
            logger.error(f"Error loading documents with DataFrameLoader: {e}")
            # Fallback to manual document creation
            documents = []
            for idx, row in df.iterrows():
                try:
                    doc = Document(
                        page_content=str(row[content_column]),
                        metadata={"source": f"doc_{idx}", "ID": str(row["ID"])}
                    )
                    documents.append(doc)
                except Exception as inner_e:
                    logger.warning(f"Skipping document at index {idx} due to error: {inner_e}")
        
        if not documents:
            logger.warning("No documents created. Check data format.")
            # Create a minimal document to initialize the vectorstore
            empty_doc = Document(page_content="Empty document placeholder", metadata={"source": "empty"})
            return FAISS.from_documents([empty_doc], embedding_model, distance_strategy=DistanceStrategy.COSINE)
                
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        try:
            document_chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(document_chunks)} chunks from {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            # Fallback to using the documents without chunking
            document_chunks = documents
            logger.info(f"Using {len(document_chunks)} documents without chunking")
        
        # Create vectorstore with proper error handling
        try:
            vectorstore_db = FAISS.from_documents(document_chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE)
            logger.info(f"Successfully created vectorstore with {len(document_chunks)} chunks")
            return vectorstore_db
        except Exception as e:
            logger.error(f"Error creating vectorstore with COSINE distance: {e}")
            try:
                # Try without specifying distance strategy
                vectorstore_db = FAISS.from_documents(document_chunks, embedding_model)
                logger.info(f"Created vectorstore without distance strategy")
                return vectorstore_db
            except Exception as inner_e:
                logger.error(f"Error creating vectorstore without distance strategy: {inner_e}")
                # Create a minimal vectorstore with an empty document as last resort
                empty_doc = Document(page_content="Empty document placeholder", metadata={"source": "empty"})
                return FAISS.from_documents([empty_doc], embedding_model)
    except Exception as e:
        logger.error(f"Unhandled error in ingest function: {e}")
        # Last resort fallback
        empty_doc = Document(page_content="Empty document placeholder", metadata={"source": "empty"})
        return FAISS.from_documents([empty_doc], embedding_model)


def process_and_ingest_files(files, embedding_model, file_type="csv"):
    """
    Process files and create a vectorstore
    
    Args:
        files: List of file paths or file-like objects
        embedding_model: Model to use for embeddings
        file_type: Type of files ('csv', 'xlsx', 'json', etc.)
        
    Returns:
        Tuple of (vectorstore, dataframe)
    """
    all_data = []
    
    for file in files:
        try:
            if file_type.lower() == "csv":
                if hasattr(file, 'read'):  # File-like object
                    df = pd.read_csv(file)
                else:  # File path
                    df = pd.read_csv(file)
            elif file_type.lower() in ["excel", "xlsx", "xls"]:
                if hasattr(file, 'read'):
                    df = pd.read_excel(file)
                else:
                    df = pd.read_excel(file)
            elif file_type.lower() == "json":
                if hasattr(file, 'read'):
                    df = pd.read_json(file)
                else:
                    df = pd.read_json(file)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                continue
                
            # Ensure ID column exists
            if "ID" not in df.columns:
                df["ID"] = range(len(df))
                
            all_data.append(df)
        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
    
    if all_data:
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create vectorstore
        vectorstore = ingest(combined_df, "Resume", embedding_model)
        
        return vectorstore, combined_df
    else:
        # Return an empty vectorstore and dataframe
        empty_doc = Document(page_content="Empty document placeholder", metadata={"source": "empty"})
        empty_vectorstore = FAISS.from_documents([empty_doc], embedding_model)
        return empty_vectorstore, pd.DataFrame(columns=["ID", "Resume"])


def save_vectorstore(vectorstore, path=FAISS_PATH):
    """
    Save the vectorstore to disk with error handling
    
    Args:
        vectorstore: FAISS vectorstore
        path: Path to save the vectorstore
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the vectorstore
        vectorstore.save_local(path)
        logger.info(f"Vectorstore saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving vectorstore: {e}")
        return False


def load_vectorstore(path=FAISS_PATH, embedding_model=None):
    """
    Load the vectorstore from disk with error handling
    
    Args:
        path: Path to load the vectorstore from
        embedding_model: Model to use for embeddings
        
    Returns:
        FAISS vectorstore or None if loading fails
    """
    try:
        if os.path.exists(path):
            vectorstore = FAISS.load_local(path, embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
            logger.info(f"Vectorstore loaded from {path}")
            return vectorstore
        else:
            logger.warning(f"Vectorstore not found at {path}")
            return None
    except Exception as e:
        logger.error(f"Error loading vectorstore: {e}")
        try:
            # Try without distance strategy
            vectorstore = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
            logger.info(f"Vectorstore loaded without distance strategy")
            return vectorstore
        except Exception as inner_e:
            logger.error(f"Error loading vectorstore without distance strategy: {inner_e}")
            return None