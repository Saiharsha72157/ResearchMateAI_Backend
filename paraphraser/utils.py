# Backend/paraphraser/utils.py

import re
from typing import List

class ParaphraseUtils:
    """
    Houses mathematical text helpers, syllable solvers, Flesch Reading Ease indices, 
    and Jaccard/LCS diversity estimators.
    """

    @staticmethod
    def count_syllables_word(word: str) -> int:
        """Counts syllables in a single word using robust english rules."""
        word = word.lower().strip()
        if not word:
            return 0
        word = re.sub(r"[^a-z]", "", word)
        if not word:
            return 0
            
        if word in ["me", "be", "he", "she", "we", "ye", "the"]:
            return 1
            
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
            
        if word.endswith("e"):
            if len(word) >= 3 and word[-2] == "l" and word[-3] not in vowels:
                pass
            else:
                count -= 1
                
        return max(1, count)

    @staticmethod
    def calculate_readability(text: str) -> float:
        """
        Calculates standard Flesch Reading Ease score between 0.0 and 1.0.
        Higher readability means simpler text.
        """
        # Sentence splitting
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        num_sentences = len(sentences)
        
        # Word extraction
        words = [w for w in re.findall(r"\b\w+\b", text)]
        num_words = len(words)
        
        if num_sentences == 0 or num_words == 0:
            return 1.0
            
        total_syllables = sum(ParaphraseUtils.count_syllables_word(w) for w in words)
        
        avg_sentence_len = num_words / num_sentences
        avg_syllables_per_word = total_syllables / num_words
        
        # Flesch formula: 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables_per_word
        fre_score = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables_per_word)
        
        # Normalize score to 0.0 - 1.0 range
        normalized = max(0.0, min(100.0, fre_score)) / 100.0
        return float(normalized)

    @staticmethod
    def calculate_diversity(original: str, candidate: str) -> float:
        """
        Computes the vocabulary diversity relative to the original text.
        Returns a float between 0.0 (no diversity / identical words) and 1.0 (highly diverse).
        Uses Jaccard Distance: 1.0 - (Intersection / Union)
        """
        orig_words = set(re.findall(r"\b\w+\b", original.lower()))
        cand_words = set(re.findall(r"\b\w+\b", candidate.lower()))
        
        if not orig_words or not cand_words:
            return 1.0
            
        intersection = orig_words.intersection(cand_words)
        union = orig_words.union(cand_words)
        
        jaccard_similarity = len(intersection) / len(union)
        jaccard_distance = 1.0 - jaccard_similarity
        
        # Add basic character length change modifier
        orig_len = len(original)
        cand_len = len(candidate)
        length_diff_modifier = min(0.1, abs(orig_len - cand_len) / max(orig_len, cand_len))
        
        diversity = jaccard_distance + length_diff_modifier
        return float(max(0.0, min(1.0, diversity)))
