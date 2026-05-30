# Backend/paraphraser/grammar_engine.py

import re

try:
    import language_tool_python
    HAS_LANGUAGE_TOOL = True
except ImportError:
    HAS_LANGUAGE_TOOL = False

class GrammarEngine:
    """
    Modular Grammar checking and correction engine.
    Utilizes local language-tool-python processes, backed by a robust
    pure-Python regex-based spell & spacing correction fallback to ensure
    crashes never occur if Java JRE is missing.
    Employing lazy-loading to completely isolate startups from crashes.
    """
    def __init__(self):
        self.tool = None
        self._init_failed = False

    def _lazy_init(self):
        """Spawns the LanguageTool JVM process on-demand during the first API query."""
        if self.tool is not None or self._init_failed:
            return
            
        if HAS_LANGUAGE_TOOL:
            try:
                print("[GrammarEngine] Spawning local LanguageTool en-US server lazily...")
                # Note: Might download language-tool files if missing, handles JRE check internally
                self.tool = language_tool_python.LanguageTool('en-US')
                print("[GrammarEngine] LanguageTool process initialized successfully!")
            except Exception as e:
                print(f"[GrammarEngine] Warning: LanguageTool server failed: {e}. Engaging pure-Python fallback.")
                self.tool = None
                self._init_failed = True
        else:
            self._init_failed = True

    def correct_grammar(self, text: str) -> str:
        """Applies spelling, spacing, capitalization, and grammar corrections to text."""
        trimmed = text.strip()
        if not trimmed:
            return ""
            
        # Trigger lazy loading checks
        self._lazy_init()
        
        # Try LanguageTool API correction
        if self.tool is not None:
            try:
                corrected = self.tool.correct(trimmed)
                if corrected and corrected.strip():
                    return corrected.strip()
            except Exception as e:
                print(f"[GrammarEngine] Correction crash: {e}. Falling back to internal heuristics.")

        # Robust Heuristic Spacing, Capitalization, and Article Correction fallback
        return self._heuristic_grammar_fix(trimmed)

    def _heuristic_grammar_fix(self, text: str) -> str:
        """Applies advanced regular expression edits to clean spacing, articles, and capitalization."""
        # 1. Spacing: space before punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        
        # 2. Spacing: double spaces collapse
        text = re.sub(r"[^\S\r\n]{2,}", " ", text)
        
        # 3. Repeated words: "the the" -> "the"
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
        
        # 4. Article a/an pairings before vowels
        text = re.sub(r"\ba\s+([aeiou]\w*)\b", r"an \1", text, flags=re.IGNORECASE)
        # Exclude common false positives like uni-
        text = re.sub(r"\ban\s+(university|unicorn|one)\b", r"a \1", text, flags=re.IGNORECASE)
        
        # 5. Article a/an pairings before consonants
        text = re.sub(r"\ban\s+([^aeiouh]\w*)\b", r"a \1", text, flags=re.IGNORECASE)

        # 6. Sentence Capitalization: first word uppercase after period
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        capitalized = []
        for s in sentences:
            if s:
                # Find first alphabet letter and capitalize it
                first_letter = re.search(r"[a-zA-Z]", s)
                if first_letter:
                    idx = first_letter.start()
                    s = s[:idx] + s[idx].upper() + s[idx+1:]
                capitalized.append(s)
                
        return " ".join(capitalized)
