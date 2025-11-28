"""TF-IDF based document retrieval for RAG."""
from pathlib import Path
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentRetriever:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        self.vectors = None
        self._load_and_index()
    
    def _load_and_index(self):
        """Load documents and create TF-IDF index."""
        # Load all markdown files
        for doc_file in self.docs_dir.glob("*.md"):
            with open(doc_file, 'r') as f:
                content = f.read()
            
            # Split into chunks (by paragraph or section)
            sections = content.split('\n\n')
            for i, section in enumerate(sections):
                if section.strip():
                    self.chunks.append({
                        'id': f"{doc_file.stem}::chunk{i}",
                        'source': doc_file.name,
                        'content': section.strip()
                    })
        
        # Build TF-IDF vectors
        if self.chunks:
            texts = [c['content'] for c in self.chunks]
            self.vectors = self.vectorizer.fit_transform(texts)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most relevant chunks."""
        if not self.chunks:
            return []
        
        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include if some relevance
                results.append({
                    'id': self.chunks[idx]['id'],
                    'source': self.chunks[idx]['source'],
                    'content': self.chunks[idx]['content'],
                    'score': float(similarities[idx])
                })
        
        return results

# Global retriever instance
_retriever = None

def get_retriever() -> DocumentRetriever:
    """Get or create the global retriever."""
    global _retriever
    if _retriever is None:
        _retriever = DocumentRetriever()
    return _retriever

def retrieve_docs(query: str, top_k: int = 3) -> List[Dict]:
    """Convenience function to retrieve documents."""
    retriever = get_retriever()
    return retriever.retrieve(query, top_k)