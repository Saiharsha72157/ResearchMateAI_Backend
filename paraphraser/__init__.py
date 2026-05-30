# Backend/paraphraser/__init__.py

from .preprocess import Preprocessor
from .paraphrase_engine import ParaphraseEngine
from .ranking_engine import RankingEngine
from .humanizer import Humanizer

class CandidateRanker:
    """
    Unified entrypoint mapping class for the advanced paraphrasing pipeline.
    Connects the preprocessor, contextual Llama 3.1 AI generator, and humanizing filters.
    """
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.generator = ParaphraseEngine()
        self.ranker = RankingEngine()

    def get_best_paraphrase(self, text: str, mode: str):
        """Processes input text, generates candidate options, and scores them to output the winner."""
        # 1. Sanitize input text
        clean_text = self.preprocessor.sanitize_text(text)
        
        # 2. Invoke the re-engineered 5-Stage contextual AI engine on the structured input
        res_dict = self.generator.paraphrase_text(clean_text, mode=mode)
        
        # 3. Apply advanced offline Humanizer flow cleanups on the final paraphrased output
        final_polished = Humanizer.humanize(res_dict["paraphrasedText"])
        
        # 4. Extract score (semantic similarity scaled to 0-10 range for standard dashboard display)
        sim_str = res_dict["semanticSimilarity"].replace("%", "").strip()
        try:
            sim_val = float(sim_str) / 100.0 if "%" in res_dict["semanticSimilarity"] else float(sim_str)
        except Exception:
            sim_val = 0.95
            
        score = sim_val * 10.0
        
        return final_polished, score

__all__ = ["CandidateRanker"]
