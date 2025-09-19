"""RAG-lite implementation for enhanced content generation."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .chunking import TextChunk
from .config import RAGConfig

logger = logging.getLogger(__name__)


class RAGManager:
    """Manages RAG-lite functionality for enhanced content generation."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.index = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        
        if config.enabled:
            self._initialize_rag_provider()
    
    def _initialize_rag_provider(self) -> None:
        """Initialize the RAG provider based on configuration."""
        if self.config.provider == "faiss":
            try:
                import faiss
                self.index = faiss.IndexFlatIP(1536)  # Assuming OpenAI embeddings dimension
                logger.info("Initialized FAISS index")
            except ImportError:
                logger.warning("FAISS not available, RAG disabled")
                self.config.enabled = False
        
        elif self.config.provider == "chroma":
            try:
                import chromadb
                # Initialize ChromaDB client
                self.chroma_client = chromadb.Client()
                self.collection = self.chroma_client.create_collection(
                    name="pdf2anki_chunks",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Initialized ChromaDB collection")
            except ImportError:
                logger.warning("ChromaDB not available, RAG disabled")
                self.config.enabled = False
        
        else:
            logger.warning(f"Unknown RAG provider: {self.config.provider}")
            self.config.enabled = False
    
    def build_index(self, chunks: List[TextChunk]) -> None:
        """Build the RAG index from text chunks."""
        if not self.config.enabled or not chunks:
            return
        
        logger.info(f"Building RAG index from {len(chunks)} chunks")
        
        # Extract embeddings for all chunks
        embeddings = []
        metadata = []
        
        for i, chunk in enumerate(chunks):
            try:
                embedding = self._get_embedding(chunk.text)
                if embedding is not None:
                    embeddings.append(embedding)
                    metadata.append({
                        "chunk_index": i,
                        "text": chunk.text,
                        "section": chunk.section,
                        "page_start": chunk.start_page,
                        "page_end": chunk.end_page,
                        "token_count": chunk.token_count,
                    })
            except Exception as e:
                logger.debug(f"Failed to get embedding for chunk {i}: {e}")
                continue
        
        if not embeddings:
            logger.warning("No embeddings generated, RAG index will be empty")
            return
        
        self.chunks_metadata = metadata
        
        if self.config.provider == "faiss":
            self._build_faiss_index(embeddings)
        elif self.config.provider == "chroma":
            self._build_chroma_index(embeddings, metadata)
        
        logger.info(f"Built RAG index with {len(embeddings)} embeddings")
    
    def _build_faiss_index(self, embeddings: List[np.ndarray]) -> None:
        """Build FAISS index from embeddings."""
        if not embeddings:
            return
        
        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / norms
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        self.embeddings = embeddings_array
    
    def _build_chroma_index(self, embeddings: List[np.ndarray], metadata: List[Dict]) -> None:
        """Build ChromaDB index from embeddings."""
        if not embeddings:
            return
        
        # Add documents to ChromaDB
        documents = [meta["text"] for meta in metadata]
        ids = [f"chunk_{meta['chunk_index']}" for meta in metadata]
        metadatas = [{k: v for k, v in meta.items() if k != "text"} for meta in metadata]
        
        self.collection.add(
            documents=documents,
            embeddings=[emb.tolist() for emb in embeddings],
            metadatas=metadatas,
            ids=ids
        )
    
    def get_relevant_chunks(self, query_text: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get top-k most relevant chunks for a query."""
        if not self.config.enabled or not self.chunks_metadata:
            return []
        
        k = k or self.config.k
        
        try:
            query_embedding = self._get_embedding(query_text)
            if query_embedding is None:
                return []
            
            if self.config.provider == "faiss":
                return self._search_faiss(query_embedding, k)
            elif self.config.provider == "chroma":
                return self._search_chroma(query_text, k)
            
        except Exception as e:
            logger.debug(f"RAG search failed: {e}")
            return []
        
        return []
    
    def _search_faiss(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search FAISS index for similar chunks."""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks_metadata):
                result = self.chunks_metadata[idx].copy()
                result["similarity_score"] = float(score)
                results.append(result)
        
        return results
    
    def _search_chroma(self, query_text: str, k: int) -> List[Dict[str, Any]]:
        """Search ChromaDB for similar chunks."""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k
            )
            
            formatted_results = []
            if results["documents"] and results["metadatas"] and results["distances"]:
                for doc, meta, distance in zip(
                    results["documents"][0], 
                    results["metadatas"][0], 
                    results["distances"][0]
                ):
                    result = meta.copy()
                    result["text"] = doc
                    result["similarity_score"] = 1 - distance  # Convert distance to similarity
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.debug(f"ChromaDB search failed: {e}")
            return []
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text using configured embedding model."""
        try:
            # This would require OpenAI API - implement as needed
            # For now, return None to disable RAG functionality
            return None
            
            # Placeholder for actual embedding implementation:
            # from openai import OpenAI
            # client = OpenAI()
            # response = client.embeddings.create(
            #     model=self.config.embedding_model,
            #     input=text
            # )
            # return np.array(response.data[0].embedding)
            
        except Exception as e:
            logger.debug(f"Failed to get embedding: {e}")
            return None
    
    def enhance_chunk_context(self, chunk: TextChunk, max_context_chunks: int = 3) -> str:
        """Enhance a chunk with relevant context from other chunks."""
        if not self.config.enabled:
            return chunk.text
        
        # Get relevant chunks
        relevant_chunks = self.get_relevant_chunks(chunk.text, k=max_context_chunks)
        
        if not relevant_chunks:
            return chunk.text
        
        # Build context
        context_parts = []
        for rel_chunk in relevant_chunks:
            if rel_chunk.get("similarity_score", 0) > 0.7:  # Only high-similarity chunks
                context_parts.append(f"[Context from p.{rel_chunk['page_start']}-{rel_chunk['page_end']}]: {rel_chunk['text'][:200]}...")
        
        if context_parts:
            enhanced_text = f"{chunk.text}\n\n--- Additional Context ---\n" + "\n\n".join(context_parts)
            return enhanced_text
        
        return chunk.text
    
    def save_index(self, index_path: Path) -> None:
        """Save the RAG index to disk."""
        if not self.config.enabled or not self.chunks_metadata:
            return
        
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.provider == "faiss" and self.index is not None:
                # Save FAISS index and metadata
                import faiss
                faiss.write_index(self.index, str(index_path / "faiss.index"))
                
                with open(index_path / "metadata.pkl", "wb") as f:
                    pickle.dump(self.chunks_metadata, f)
                
            elif self.config.provider == "chroma":
                # ChromaDB persists automatically if using persistent client
                # For in-memory client, we'd need to export/import
                logger.debug("ChromaDB index persistence handled by ChromaDB")
            
            logger.info(f"Saved RAG index to {index_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save RAG index: {e}")
    
    def load_index(self, index_path: Path) -> None:
        """Load the RAG index from disk."""
        if not self.config.enabled:
            return
        
        try:
            if self.config.provider == "faiss":
                faiss_path = index_path / "faiss.index"
                metadata_path = index_path / "metadata.pkl"
                
                if faiss_path.exists() and metadata_path.exists():
                    import faiss
                    self.index = faiss.read_index(str(faiss_path))
                    
                    with open(metadata_path, "rb") as f:
                        self.chunks_metadata = pickle.load(f)
                    
                    logger.info(f"Loaded RAG index with {len(self.chunks_metadata)} chunks")
            
        except Exception as e:
            logger.warning(f"Failed to load RAG index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG statistics."""
        return {
            "enabled": self.config.enabled,
            "provider": self.config.provider,
            "embedding_model": self.config.embedding_model,
            "k": self.config.k,
            "chunks_indexed": len(self.chunks_metadata),
            "index_size": self.index.ntotal if self.config.provider == "faiss" and self.index else 0,
        }


def create_rag_manager(config: RAGConfig) -> RAGManager:
    """Factory function to create a RAG manager."""
    return RAGManager(config)