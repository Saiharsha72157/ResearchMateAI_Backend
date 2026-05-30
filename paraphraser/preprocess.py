# Backend/paraphraser/preprocess.py

import re
from typing import List, Dict, Any

class Preprocessor:
    """
    Standard preprocessor for cleaning, normalizing, segmenting,
    and validating text before paraphrasing.
    """
    def __init__(self, max_words: int = 200):
        self.max_words = max_words
        self.sentence_end_regex = re.compile(r"(?<=[.!?])\s+")

    def normalize_punctuation(self, text: str) -> str:
        """Standardizes smart quotes, typography characters, and odd spaces."""
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("—", "-").replace("–", "-")
        return text

    def remove_extra_spaces(self, text: str) -> str:
        """Collapses duplicate spaces, tabs, and double lines."""
        return re.sub(r"[^\S\r\n]{2,}", " ", text).strip()

    def segment_sentences(self, text: str) -> List[str]:
        """Segments text paragraph into individual clean sentences."""
        clean_text = self.remove_extra_spaces(self.normalize_punctuation(text))
        sentences = self.sentence_end_regex.split(clean_text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        """Splits sentence string into clean word tokens without punctuation."""
        return [w for w in re.findall(r"\b\w+\b", text.lower())]

    def validate_input(self, text: str) -> Dict[str, Any]:
        """Verifies if input fits constraints and max word counts."""
        trimmed = text.strip()
        if not trimmed:
            return {"valid": False, "error": "Text content cannot be empty."}
            
        words = trimmed.split()
        word_count = len(words)
        if word_count > self.max_words:
            return {
                "valid": False, 
                "error": f"Maximum {self.max_words} words allowed (detected {word_count} words). Please trim your text."
            }
            
        return {"valid": True, "word_count": word_count}

    def sanitize_text(self, text: str) -> str:
        """Executes full sanitization pipeline on the text string."""
        return self.remove_extra_spaces(self.normalize_punctuation(text))
