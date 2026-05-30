# Backend/paraphraser/semantic_engine.py

import re
import math
from typing import List

try:
    import torch
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class SemanticEngine:
    """
    Computes semantic similarity embeddings and cosine scores between text strings.
    Employs HuggingFace SentenceTransformers internally, with a direct 
    pure-Python TF-IDF cosine similarity fallback.
    Employing lazy-loading to completely isolate startups from crashes.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._init_failed = False

    def _lazy_init(self):
        """Loads SentenceTransformer weights on-demand during the first API query."""
        if self.model is not None or self._init_failed:
            return
            
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                print(f"[SemanticEngine] Initializing sentence-transformer '{self.model_name}' lazily...")
                # Download/load model with a short local fallback / warning
                self.model = SentenceTransformer(self.model_name)
                print("[SemanticEngine] Model weights loaded successfully!")
            except Exception as e:
                print(f"[SemanticEngine] Warning: Transformer weights failed to load: {e}. Engaging TF-IDF fallback.")
                self.model = None
                self._init_failed = True
        else:
            self._init_failed = True

    def calculate_similarity(self, original: str, candidate: str) -> float:
        """
        Computes semantic similarity score between original and candidate.
        Returns a float between 0.0 and 1.0.
        """
        orig_clean = original.strip().lower()
        cand_clean = candidate.strip().lower()
        
        if not orig_clean or not cand_clean:
            return 0.0
            
        if orig_clean == cand_clean:
            return 1.0

        # Trigger lazy loading checks
        self._lazy_init()

        # Try HuggingFace Sentence-Transformers Cosine Similarity
        if self.model is not None:
            try:
                emb1 = self.model.encode(original, convert_to_tensor=True)
                emb2 = self.model.encode(candidate, convert_to_tensor=True)
                cos_sim = util.cos_sim(emb1, emb2)
                return float(cos_sim.item())
            except Exception as e:
                print(f"[SemanticEngine] Encoding error: {e}. Falling back to TF-IDF cosine comparison.")
                
        # Pure Python TF-IDF Cosine Similarity fallback
        return self._pure_tfidf_similarity(orig_clean, cand_clean)

    def _pure_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Robust TF-IDF / Bag of Words Cosine Similarity backup solver."""
        w_regex = re.compile(r"\b\w+\b")
        words1 = w_regex.findall(text1)
        words2 = w_regex.findall(text2)
        
        if not words1 or not words2:
            return 0.0
            
        # Compile joint vocabulary
        vocab = set(words1).union(set(words2))
        
        # Word frequency vectors
        vec1 = {w: 0 for w in vocab}
        vec2 = {w: 0 for w in vocab}
        
        for w in words1:
            vec1[w] += 1
        for w in words2:
            vec2[w] += 1
            
        # Cosine dot-product math
        dot_product = sum(vec1[w] * vec2[w] for w in vocab)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
            
        similarity = dot_product / (mag1 * mag2)
        
        # Add a light word-overlap Jaccard buffer for synonyms or structural shifts
        w_set1 = set(words1)
        w_set2 = set(words2)
        jaccard = len(w_set1.intersection(w_set2)) / len(w_set1.union(w_set2))
        
        # Mix cosine with overlap buffer
        final_sim = (0.7 * similarity) + (0.3 * jaccard)
        return float(max(0.0, min(1.0, final_sim)))
