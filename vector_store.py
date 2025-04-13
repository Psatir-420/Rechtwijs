import os
import json
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("VectorStore")

class VectorStore:
    def __init__(self, data_dir="processed_data"):
        """Initialize the vector store using sklearn instead of FAISS.
        
        Args:
            data_dir (str): Directory containing processed PDF data
        """
        self.data_dir = data_dir
        self.documents = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            token_pattern=r'\b\w+\b'  # Only consider words as tokens
        )
        self.vectors = None
        self.chunk_texts = []
        self.chunk_metadata = []
        
        logger.info(f"Initialized vector store with data directory: {data_dir}")
    
    def load_documents(self):
        """Load processed documents from the data directory."""
        try:
            self.documents = []
            document_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
            
            if not document_files:
                logger.warning(f"No document files found in {self.data_dir}")
                return
            
            logger.info(f"Loading {len(document_files)} documents from {self.data_dir}")
            
            for file in document_files:
                file_path = os.path.join(self.data_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        document = json.load(f)
                        self.documents.append(document)
                except Exception as e:
                    logger.error(f"Error loading document {file_path}: {str(e)}")
            
            # Prepare for vectorization
            self._prepare_vectors()
            
            logger.info(f"Successfully loaded {len(self.documents)} documents with {len(self.chunk_texts)} chunks")
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
    
    def _prepare_vectors(self):
        """Prepare vectors for similarity search using sklearn."""
        # Reset
        self.chunk_texts = []
        self.chunk_metadata = []
        
        # Extract all chunks from documents
        for doc in self.documents:
            for chunk in doc["chunks"]:
                self.chunk_texts.append(chunk["text"])
                
                # Store metadata for retrieval
                self.chunk_metadata.append({
                    "source": doc["source"],
                    "metadata": chunk["metadata"],
                    "text": chunk["text"]
                })
        
        # Skip vectorization if no chunks
        if not self.chunk_texts:
            logger.warning("No chunks to vectorize")
            return
        
        # Vectorize chunks
        try:
            # Create and fit the TF-IDF vectorizer
            self.vectors = self.vectorizer.fit_transform(self.chunk_texts)
            logger.info(f"Vectorized {len(self.chunk_texts)} chunks")
        except Exception as e:
            logger.error(f"Error vectorizing text: {str(e)}")
            self.vectors = None
    
    def similarity_search(self, query, top_k=5):
        """Perform a similarity search using cosine similarity.
        
        Args:
            query (str): Query text
            top_k (int): Number of most similar results to return
            
        Returns:
            list: List of most similar chunks with their metadata
        """
        if self.vectors is None or not self.chunk_texts:
            logger.warning("No vectors available for search")
            return []
        
        try:
            # Vectorize the query
            query_vec = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vec, self.vectors)[0]
            
            # Get indices of top k most similar documents
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Return results
            results = []
            for idx in top_indices:
                results.append({
                    "score": float(similarities[idx]),
                    "source": self.chunk_metadata[idx]["source"],
                    "metadata": self.chunk_metadata[idx]["metadata"],
                    "text": self.chunk_metadata[idx]["text"]
                })
            
            logger.info(f"Found {len(results)} relevant chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
