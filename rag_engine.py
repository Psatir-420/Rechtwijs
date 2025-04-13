import os
import logging
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RAGEngine")

class RAGEngine:
    def __init__(self, vector_store, api_key):
        """Initialize the RAG engine.
        
        Args:
            vector_store (VectorStore): Vector store for document retrieval
            api_key (str): Gemini API key
        """
        self.vector_store = vector_store
        self.api_key = api_key
        self.model_name = "gemini-2.0-flash-thinking-exp-01-21"
        
        # Initialize Gemini client
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("Initialized Gemini client")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {str(e)}")
            self.model = None
    
    def generate_response(self, query, num_results=3):
        """Generate a response using RAG.
        
        Args:
            query (str): User query
            num_results (int): Number of documents to retrieve
            
        Returns:
            dict: Response with answer and sources
        """
        if not self.model:
            logger.error("Gemini model not initialized")
            return {
                "answer": "Error: Gemini API model not initialized. Please check your API key.",
                "sources": []
            }
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query, top_k=num_results)
            
            if not relevant_docs:
                logger.warning("No relevant documents found for query")
                return {
                    "answer": "I couldn't find any relevant information to answer your question. Please try a different query or make sure documents have been processed.",
                    "sources": []
                }
            
            # Create context from retrieved documents
            context = self._create_context(relevant_docs)
            
            # Create prompt with context
            prompt = self._create_prompt(query, context)
            
            # Generate response using Gemini
            response = self._generate_with_gemini(prompt)
            
            logger.info(f"Generated response for query: {query[:50]}...")
            
            return {
                "answer": response,
                "sources": relevant_docs
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": []
            }
    
    def _create_context(self, relevant_docs):
        """Create context from relevant documents.
        
        Args:
            relevant_docs (list): List of relevant document chunks
            
        Returns:
            str: Context text
        """
        context = "Here are some relevant documents to help answer the question:\n\n"
        
        for i, doc in enumerate(relevant_docs):
            context += f"Document {i+1} (Source: {os.path.basename(doc['source'])}, Pages: {doc['metadata']['page_start']}-{doc['metadata']['page_end']}):\n"
            context += doc["text"] + "\n\n"
        
        return context
    
    def _create_prompt(self, query, context):
    """Membuat prompt untuk model Gemini dalam Bahasa Indonesia.

    Args:
        query (str): Pertanyaan pengguna
        context (str): Konteks hasil pencarian dokumen

    Returns:
        str: Prompt lengkap dalam Bahasa Indonesia
    """
    prompt = f"""Anda adalah asisten hukum yang ahli dalam hukum Indonesia. Berdasarkan dokumen-dokumen berikut, berikan jawaban yang relevan, jelas, dan mudah dipahami.

{context}

Silakan jawab pertanyaan di bawah ini berdasarkan informasi yang terdapat dalam dokumen di atas. Anda boleh menyusun ulang kalimat dengan bahasa Anda sendiri selama maknanya tetap sesuai dengan dokumen. Jangan menambahkan informasi dari luar dokumen. Jika dokumen tidak memuat informasi yang cukup, sampaikan bahwa jawabannya tidak tersedia.

Question: {query}

Answer:"""
    
    return prompt

    
    def _generate_with_gemini(self, prompt):
        """Generate a response using the Gemini API.
        
        Args:
            prompt (str): Complete prompt
            
        Returns:
            str: Generated response
        """
        try:
            # Generate content with the model
            response = self.model.generate_content(prompt)
            
            # Extract text from response
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in Gemini generation: {str(e)}")
            return f"Error generating response: {str(e)}"
