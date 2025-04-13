import os
import json
import logging
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("VectorStore")

class VectorStore:
    def __init__(self, data_dir="processed_data"):
        """Initialize the vector store.
        
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
        self.index = None
        self.chunk_texts = []
        self.chunk_metadata = []
        self.dimension = 5000  # Same as max_features in TfidfVectorizer
        
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
        """Prepare vectors for similarity search using FAISS."""
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
            sparse_vectors = self.vectorizer.fit_transform(self.chunk_texts)
            
            # Convert sparse vectors to dense for FAISS
            dense_vectors = sparse_vectors.toarray().astype('float32')
            
            # Create FAISS index - using L2 distance
            self.dimension = dense_vectors.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # Add vectors to the index
            self.index.add(dense_vectors)
            
            logger.info(f"Vectorized {len(self.chunk_texts)} chunks and built FAISS index")
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            self.index = None
    
    def similarity_search(self, query, top_k=5):
        """Perform a similarity search using FAISS.
        
        Args:
            query (str): Query text
            top_k (int): Number of most similar results to return
            
        Returns:
            list: List of most similar chunks with their metadata
        """
        if self.index is None or not self.chunk_texts:
            logger.warning("No vectors available for search")
            return []
        
        try:
            # Vectorize the query
            query_vec = self.vectorizer.transform([query])
            query_dense = query_vec.toarray().astype('float32')
            
            # Perform search
            distances, indices = self.index.search(query_dense, min(top_k, len(self.chunk_texts)))
            
            # Return results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.chunk_metadata):
                    continue  # Skip invalid indices
                
                # Lower distance is better in L2 space, so convert to similarity score
                # by using a simple inverse (1/1+distance)
                similarity = 1.0 / (1.0 + distances[0][i])
                
                results.append({
                    "score": float(similarity),
                    "source": self.chunk_metadata[idx]["source"],
                    "metadata": self.chunk_metadata[idx]["metadata"],
                    "text": self.chunk_metadata[idx]["text"]
                })
            
            logger.info(f"Found {len(results)} relevant chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
