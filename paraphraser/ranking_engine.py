# Backend/paraphraser/ranking_engine.py

from typing import List, Tuple
from paraphraser.preprocess import Preprocessor
from paraphraser.modes import WritingModes
from paraphraser.utils import ParaphraseUtils
from paraphraser.semantic_engine import SemanticEngine
from paraphraser.grammar_engine import GrammarEngine
from paraphraser.writing_analyzer import WritingAnalyzer

class RankingEngine:
    """
    Advanced candidate evaluation and scoring engine.
    Applies the formula: 
      Final Score = 0.40 * SemanticSimilarity + 0.30 * GrammarScore + 0.20 * ReadabilityScore + 0.10 * DiversityScore
    to elect the highest-quality paraphrase output.
    """
    def __init__(self):
        self.semantic_engine = SemanticEngine()
        self.grammar_engine = GrammarEngine()
        self.grammar_analyzer = WritingAnalyzer()

    def _calculate_grammar_score(self, text: str) -> float:
        """Evaluates grammar cleanliness and returns a score between 0.0 (poor) and 1.0 (flawless)."""
        try:
            issues = self.grammar_analyzer.analyze_grammar(text)
            # Deduct 0.1 for every grammar issue flagged, minimum 0.0
            return max(0.0, min(1.0, 1.0 - (0.1 * len(issues))))
        except Exception:
            return 1.0

    def rank_sentence_candidates(self, original: str, candidates: List[str], mode: str) -> Tuple[str, float]:
        """
        Grades and ranks sentence-level candidates. Returns the highest-scoring candidate
        sentence and its raw score (0.0 to 1.0).
        """
        weights = WritingModes.get_weights(mode)
        
        best_candidate = original
        best_score = 0.0
        
        orig_norm = original.strip().lower().rstrip(".!?")
        
        for cand in candidates:
            cand_text = cand.strip()
            if not cand_text:
                continue
                
            # Skip identical candidates to guarantee the output is actually paraphrased
            cand_norm = cand_text.lower().rstrip(".!?")
            if cand_norm == orig_norm:
                continue
                
            # 1. Compute Semantic Similarity Score (0.0 to 1.0)
            sem_sim = self.semantic_engine.calculate_similarity(original, cand_text)
            
            # 2. Compute Grammar cleanliness Score (0.0 to 1.0)
            gram_score = self._calculate_grammar_score(cand_text)
            
            # 3. Compute Readability score (0.0 to 1.0)
            read_score = ParaphraseUtils.calculate_readability(cand_text)
            
            # 4. Compute Diversity relative to original (0.0 to 1.0)
            div_score = ParaphraseUtils.calculate_diversity(original, cand_text)
            
            # Weighted average
            score = (
                (weights["semantic"] * sem_sim) +
                (weights["grammar"] * gram_score) +
                (weights["readability"] * read_score) +
                (weights["diversity"] * div_score)
            )
            
            # Track highest-scoring candidate
            if score > best_score:
                best_score = score
                best_candidate = cand_text
                
        # Ensure we run a final grammar fix on the winning candidate sentence
        final_polished = self.grammar_engine.correct_grammar(best_candidate)
        
        return final_polished, best_score

    def rank_candidates(self, original: str, candidates: List[str], mode: str) -> Tuple[str, float]:
        """
        Evaluates and ranks candidate paragraphs based on mode-specific scoring weights.
        Returns the highest-scoring candidate and its final scaled score.
        """
        weights = WritingModes.get_weights(mode)
        
        best_candidate = original
        best_score = 0.0
        
        orig_norm = original.strip().lower().rstrip(".!?")
        
        for cand in candidates:
            cand_text = cand.strip()
            if not cand_text:
                continue
                
            # Skip identical candidates to guarantee the output is actually paraphrased
            cand_norm = cand_text.lower().rstrip(".!?")
            if cand_norm == orig_norm:
                continue
                
            # 1. Compute Semantic Similarity Score (0.0 to 1.0)
            sem_sim = self.semantic_engine.calculate_similarity(original, cand_text)
            
            # 2. Compute Grammar cleanliness Score (0.0 to 1.0)
            gram_score = self._calculate_grammar_score(cand_text)
            
            # 3. Compute Readability score (0.0 to 1.0)
            read_score = ParaphraseUtils.calculate_readability(cand_text)
            
            # 4. Compute Diversity relative to original (0.0 to 1.0)
            div_score = ParaphraseUtils.calculate_diversity(original, cand_text)
            
            # Weighted average
            score = (
                (weights["semantic"] * sem_sim) +
                (weights["grammar"] * gram_score) +
                (weights["readability"] * read_score) +
                (weights["diversity"] * div_score)
            )
            
            # Track highest-scoring candidate
            if score > best_score:
                best_score = score
                best_candidate = cand_text
                
        # Scale score to 0.0 - 10.0 range for premium QuillBot dashboard display
        scaled_score = round(max(0.0, min(10.0, best_score * 10.0)), 1)
        
        # Ensure we run a final grammar fix on the winning candidate
        final_polished = self.grammar_engine.correct_grammar(best_candidate)
        
        return final_polished, scaled_score
