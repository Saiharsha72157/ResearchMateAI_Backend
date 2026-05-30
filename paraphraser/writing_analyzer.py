# Backend/paraphraser/writing_analyzer.py

import re
from typing import List, Dict, Any

class WritingAnalyzer:
    """
    Intelligent offline writing analysis engine.
    Computes grammar checking, readability analysis, tone detection, and stylistic suggestions.
    Does not require external APIs or bulky NLP libraries.
    """

    def __init__(self):
        # Curated lexicons for tone detection
        self.tone_lexicons = {
            "academic": {
                "hypothesize", "hypothesized", "demonstrate", "demonstrates", "demonstrated", 
                "analyze", "analyzed", "analyzing", "investigate", "investigated", "investigating", 
                "correlation", "methodology", "framework", "empirical", "subsequent", "consequently", 
                "theory", "theoretical", "literature", "significance", "significant", "evaluate", 
                "evaluated", "evaluating", "synthesize", "synthesized", "synthesizing", "phenomenon", 
                "paradigm", "postulate", "validate", "validity", "quantify", "quantitative", "qualitative"
            },
            "formal": {
                "appreciate", "appreciated", "establish", "established", "request", "requested", 
                "regarding", "furthermore", "additional", "observe", "observed", "conduct", "conducted", 
                "verify", "verified", "therefore", "nevertheless", "hereby", "determine", "determined", 
                "assist", "assisted", "obtain", "obtained", "utilize", "utilized", "optimal", 
                "commence", "commenced", "subsequently", "manifest", "manifested", "hence", "moreover"
            },
            "casual": {
                "cool", "awesome", "totally", "dynamic", "super", "basically", "stuff", "guy", "guys", 
                "kid", "kids", "guess", "yeah", "hey", "wow", "lol", "anyway", "chill", "fun", 
                "gonna", "wanna", "gotta", "yolo", "superb", "crazy", "huge", "nice", "ok", "okay"
            },
            "creative": {
                "vibrant", "whisper", "whispered", "spark", "sparked", "dream", "dreamed", "dreamt", 
                "capture", "captured", "paint", "painted", "brilliant", "beautiful", "mysterious", 
                "journey", "infinite", "glowing", "embrace", "embraced", "dance", "danced", "shadow", 
                "shadows", "melancholy", "magic", "magical", "soul", "heart", "wonder", "wondered", "vivid"
            },
            "technical": {
                "algorithm", "algorithms", "database", "databases", "execution", "parameter", "parameters", 
                "configuration", "configurations", "compiler", "syntax", "endpoint", "endpoints", 
                "binary", "backend", "frontend", "architecture", "architectures", "api", "apis", 
                "variable", "variables", "function", "functions", "array", "arrays", "object-oriented", 
                "server", "servers", "latency", "bandwidth", "integration", "debug", "debugging"
            }
        }

        # Simpler word recommendations mapping
        self.simplifications = {
            "utilize": "use",
            "utilizes": "uses",
            "utilized": "used",
            "utilizing": "using",
            "facilitate": "help",
            "facilitates": "helps",
            "facilitated": "helped",
            "facilitating": "helping",
            "subsequent": "later",
            "subsequently": "later",
            "nevertheless": "but",
            "consequently": "so",
            "terminate": "end",
            "terminated": "ended",
            "commence": "start",
            "commenced": "started",
            "implement": "carry out",
            "implemented": "carried out",
            "implementing": "carrying out",
            "aggregate": "total",
            "aggregates": "totals",
            "aggregated": "totaled",
            "endeavor": "try",
            "endeavored": "tried",
            "fundamental": "basic",
            "approximately": "about"
        }

        # Passive voice markers (auxiliary verb + past participles)
        self.passive_regex = re.compile(
            r"\b(is|am|are|was|were|be|been|being)\s+(\w+ed|written|done|made|taken|given|seen|chosen|broken|told|held|built|kept|shown|found|bought|sold|sent|lost|paid|met)\b",
            re.IGNORECASE
        )

        # Stopwords to filter out when checking for word repetition
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "because", "as", "until", "while", "of", 
            "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", 
            "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", 
            "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", 
            "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", 
            "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", 
            "s", "t", "can", "will", "just", "don", "should", "shouldn", "now", "i", "me", "my", 
            "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
            "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", 
            "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", 
            "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", 
            "doing", "would", "could", "should", "ought", "might", "must", "shall"
        }

    def _count_syllables_word(self, word: str) -> int:
        """Counts the syllables in a single english word using robust heuristics."""
        word = word.lower().strip()
        if not word:
            return 0
        
        # Strip simple non-alphabetic characters
        word = re.sub(r"[^a-z]", "", word)
        if not word:
            return 0
            
        # Exception list for small words
        if word in ["me", "be", "he", "she", "we", "ye", "the"]:
            return 1
            
        # Syllables vowel matching
        vowels = "aeiouy"
        count = 0
        
        # Check vowel clusters
        prev_is_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
            
        # Strip silent trailing e
        if word.endswith("e"):
            # Words ending in -le preceded by a consonant keep the e syllable (e.g. table, candle)
            if len(word) >= 3 and word[-2] == "l" and word[-3] not in vowels:
                pass
            else:
                count -= 1
                
        # Assure at least 1 syllable per word
        return max(1, count)

    def _segment_sentences(self, text: str) -> List[str]:
        """Splits a paragraph into sentences safely using standard boundary marks."""
        # Regex boundary segmenter
        sentence_end = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_end.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def analyze_grammar(self, text: str) -> List[Dict[str, Any]]:
        """Scans the text for grammatical, punctuation, capitalization, and spacing issues."""
        issues = []
        trimmed = text.strip()
        if not trimmed:
            return issues

        # 1. Spacing check: spaces before punctuation
        for m in re.finditer(r"(\w+)\s+([.,!?;:])", text):
            issues.append({
                "type": "grammar",
                "message": f"Extra space before punctuation '{m.group(2)}'.",
                "severity": "warning",
                "position": m.start()
            })

        # 2. Spacing check: double/multiple spaces
        for m in re.finditer(r"[^\S\r\n]{2,}", text):
            issues.append({
                "type": "grammar",
                "message": "Multiple consecutive spaces detected.",
                "severity": "warning",
                "position": m.start()
            })

        # 3. Repeated contiguous words (e.g. "the the")
        for m in re.finditer(r"\b(\w+)\s+\1\b", text, re.IGNORECASE):
            issues.append({
                "type": "grammar",
                "message": f"Repeated word '{m.group(1)}' detected.",
                "severity": "warning",
                "position": m.start()
            })

        # 4. Article mismatch checker: "a" before vowels
        for m in re.finditer(r"\ba\s+([aeiou]\w*)\b", text, re.IGNORECASE):
            word_following = m.group(1).lower()
            # Exclude exceptions like university, union, euphemism, etc.
            if not (word_following.startswith("uni") or word_following.startswith("euphem") or word_following == "one"):
                issues.append({
                    "type": "grammar",
                    "message": f"Incorrect article. Use 'an' before the vowel-sounding word '{m.group(1)}'.",
                    "severity": "warning",
                    "position": m.start()
                })

        # 5. Article mismatch checker: "an" before consonants
        for m in re.finditer(r"\ban\s+([^aeiouh]\w*)\b", text, re.IGNORECASE):
            word_following = m.group(1).lower()
            # Exclude words starting with vocal consonants if any (standard consonant sounds)
            issues.append({
                "type": "grammar",
                "message": f"Incorrect article. Use 'a' before the consonant-sounding word '{m.group(1)}'.",
                "severity": "warning",
                "position": m.start()
            })

        # 6. Punctuation checker: sentences ending without standard marks
        sentences = self._segment_sentences(text)
        current_offset = 0
        for s in sentences:
            start_pos = text.find(s, current_offset)
            if start_pos != -1:
                current_offset = start_pos + len(s)
                # Verify trailing punctuation on the sentence segment
                if s and s[-1] not in ".!?":
                    issues.append({
                        "type": "grammar",
                        "message": "Sentence is missing trailing punctuation (period, question mark, or exclamation mark).",
                        "severity": "warning",
                        "position": start_pos + len(s) - 1
                    })

        # 7. Sentence Capitalization Check
        current_offset = 0
        for s in sentences:
            start_pos = text.find(s, current_offset)
            if start_pos != -1:
                current_offset = start_pos + len(s)
                first_char_match = re.search(r"[a-zA-Z]", s)
                if first_char_match:
                    first_letter = first_char_match.group()
                    first_letter_pos = start_pos + first_char_match.start()
                    if first_letter.islower():
                        issues.append({
                            "type": "grammar",
                            "message": f"Capitalization issue: Sentence starting word should begin with an uppercase letter.",
                            "severity": "warning",
                            "position": first_letter_pos
                        })

        # 8. Common Grammatical Confusions
        # "its" instead of "it's" in obvious auxiliary places
        for m in re.finditer(r"\bits\s+(?:a|an|the|very|not|going|doing|about)\b", text, re.IGNORECASE):
            issues.append({
                "type": "grammar",
                "message": "Grammar mistake: Use 'it\'s' (it is) instead of 'its' (possessive) in this context.",
                "severity": "warning",
                "position": m.start()
            })

        # "your" instead of "you're"
        for m in re.finditer(r"\byour\s+(?:going|doing|welcome|right|great)\b", text, re.IGNORECASE):
            issues.append({
                "type": "grammar",
                "message": "Grammar mistake: Use 'you\'re' (you are) instead of 'your' (possessive) in this context.",
                "severity": "warning",
                "position": m.start()
            })

        # Tense inconsistency warning: simple mix of was and is in a single sentence
        for s in sentences:
            start_pos = text.find(s)
            if "was" in s.lower() and "is" in s.lower():
                issues.append({
                    "type": "grammar",
                    "message": "Possible tense inconsistency: adjacent past ('was') and present ('is') auxiliary verbs in the same sentence.",
                    "severity": "warning",
                    "position": start_pos if start_pos != -1 else 0
                })

        return sorted(issues, key=lambda x: x["position"])

    def analyze_readability(self, text: str) -> Dict[str, Any]:
        """Computes readability scores based on sentence structures, syllables, and averages."""
        sentences = self._segment_sentences(text)
        num_sentences = len(sentences)
        
        # Filter text to count words
        words = [w for w in re.findall(r"\b\w+\b", text)]
        num_words = len(words)
        
        if num_sentences == 0 or num_words == 0:
            return {"score": 100, "level": "Easy"}
            
        # Count syllables
        total_syllables = sum(self._count_syllables_word(w) for w in words)
        
        avg_sentence_len = num_words / num_sentences
        avg_syllables_per_word = total_syllables / num_words
        
        # Flesch Reading Ease Formula
        score = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables_per_word)
        score = round(max(0, min(100, score)))
        
        # Classification mapping
        if score >= 61:
            level = "Easy"
        elif score >= 31:
            level = "Moderate"
        else:
            level = "Complex"
            
        return {
            "score": score,
            "level": level
        }

    def detect_tone(self, text: str) -> Dict[str, Any]:
        """Classifies writing tone (Academic, Formal, Casual, Creative, Technical, Neutral) with confidence."""
        words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
        if not words:
            return {"tone": "Neutral", "confidence": 100}
            
        scores = {
            "Academic": 0,
            "Formal": 0,
            "Casual": 0,
            "Creative": 0,
            "Technical": 0
        }
        
        # Word matches against our lists
        for w in words:
            for tone, lexicon in self.tone_lexicons.items():
                # Map internal keys to labels
                label = tone.capitalize()
                if w in lexicon:
                    scores[label] += 1.5  # Heavy weight for vocabulary matches
                    
        # Style heuristics
        # CASUAL: exclamation marks
        num_excl = text.count("!")
        scores["Casual"] += num_excl * 2.0
        
        # CASUAL: question marks (slightly casual or creative)
        num_q = text.count("?")
        scores["Casual"] += num_q * 0.5
        
        # ACADEMIC: passive voice constructs
        passive_matches = len(self.passive_regex.findall(text))
        scores["Academic"] += passive_matches * 1.5
        
        total_score = sum(scores.values())
        if total_score == 0:
            return {"tone": "Neutral", "confidence": 100}
            
        # Identify dominant tone
        dominant_tone = max(scores, key=scores.get)
        dominant_val = scores[dominant_tone]
        
        # Compute confidence percentage (capped to keep it premium and standard)
        confidence = round((dominant_val / total_score) * 100)
        
        # Let's enforce that if confidence is very low, it drops back to Neutral
        if confidence < 35:
            return {"tone": "Neutral", "confidence": 100}
            
        return {
            "tone": dominant_tone,
            "confidence": confidence
        }

    def generate_suggestions(self, text: str) -> List[Dict[str, Any]]:
        """Constructs style recommendations regarding passive voice, long sentences, word variety, and complex structures."""
        suggestions = []
        sentences = self._segment_sentences(text)
        
        # 1. Passive voice detection
        for m in self.passive_regex.finditer(text):
            suggestions.append({
                "type": "passive_voice",
                "message": f"Passive voice detected ('{m.group(0)}'). Consider using active voice for direct and engaging pacing."
            })
            
        # 2. Simplification mappings
        words = [w for w in re.findall(r"\b\w+\b", text)]
        seen_words = set()
        for w in words:
            wl = w.lower()
            if wl in self.simplifications and wl not in seen_words:
                seen_words.add(wl)
                suggestions.append({
                    "type": "word_choice",
                    "message": f"Consider simpler wording: Replace complex term '{w}' with '{self.simplifications[wl]}'."
                })
                
        # 3. Word Repetition Detection
        word_counts = {}
        for w in words:
            wl = w.lower()
            if len(wl) > 4 and wl not in self.stopwords:
                word_counts[wl] = word_counts.get(wl, 0) + 1
                
        for wl, count in word_counts.items():
            if count >= 3:
                suggestions.append({
                    "type": "vocabulary",
                    "message": f"Vocabulary repetition detected: Word '{wl}' was used {count} times. Consider using synonyms."
                })
                
        # 4. Long Sentence Detection
        for s in sentences:
            sentence_words = [w for w in re.findall(r"\b\w+\b", s)]
            if len(sentence_words) > 25:
                suggestions.append({
                    "type": "sentence_length",
                    "message": f"Sentence is too long ({len(sentence_words)} words). Consider dividing it to improve reader readability."
                })
                
        return suggestions
