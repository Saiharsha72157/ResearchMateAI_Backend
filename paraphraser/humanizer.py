# Backend/paraphraser/humanizer.py

import re

class Humanizer:
    """
    Applies offline humanizing heuristics to the paraphrased paragraph:
    1. Rotates repetitive sentence starts.
    2. Combines simple repetitive adjacent sentences.
    3. Compresses stacked awkward synonym adjective chains.
    4. Cleans duplicate overlapping transitions.
    """
    
    @staticmethod
    def humanize(text: str) -> str:
        if not text.strip():
            return text
            
        # Clean hyphen spacing
        text = re.sub(r"-\s+", "-", text)
        
        # 1. Clean Duplicate Overlapping Transitions (Feature 4 & 5)
        text = re.sub(r"\bconsequently,\s+therefore,?\b", "Consequently,", text, flags=re.IGNORECASE)
        text = re.sub(r"\bconsequently\s+therefore\b", "consequently", text, flags=re.IGNORECASE)
        
        # 2. Combine Repetitive Adjacent Sentences (Feature 2)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        
        combined_sentences = []
        skip_next = False
        
        for idx in range(len(sentences)):
            if skip_next:
                skip_next = False
                continue
                
            if idx < len(sentences) - 1:
                s1 = sentences[idx]
                s2 = sentences[idx + 1]
                
                # Check for "The [Noun] is [Adj]."
                m1 = re.match(r"^The\s+(\w+)\s+(is|was)\s+([\w\s-]+?)(?:\.)?$", s1, re.IGNORECASE)
                m2 = re.match(r"^The\s+(\w+)\s+(is|was)\s+([\w\s-]+?)(?:\.)?$", s2, re.IGNORECASE)
                
                if m1 and m2 and m1.group(1).lower() == m2.group(1).lower() and m1.group(2).lower() == m2.group(2).lower():
                    noun = m1.group(1)
                    verb = m1.group(2)
                    adj1 = m1.group(3).strip()
                    adj2 = m2.group(3).strip()
                    
                    # Combine!
                    if adj1 == "effective" and adj2 == "reliable":
                        combined = f"The {noun} {verb} {adj1} and also demonstrates reliability."
                    elif adj1 == "important" and adj2 == "useful":
                        combined = f"The {noun} {verb} {adj1} and useful."
                    else:
                        combined = f"The {noun} {verb} {adj1} and also {adj2}."
                    
                    combined_sentences.append(combined)
                    skip_next = True
                    continue
                    
            combined_sentences.append(sentences[idx])
            
        sentences = combined_sentences
        
        # 3. Rotate Repetitive Sentence Starts (Feature 1)
        openers = ["Additionally", "Furthermore", "Moreover", "In particular", "Consequently"]
        opener_idx = 0
        
        rotated_sentences = []
        prev_start = None
        
        for s in sentences:
            if not s:
                continue
                
            words = s.split()
            if len(words) >= 2:
                start_phrase = " ".join(words[:2]).lower().rstrip(",.")
                if prev_start and start_phrase == prev_start:
                    op = openers[opener_idx % len(openers)]
                    opener_idx += 1
                    first_word = words[0]
                    if first_word != "I":
                        first_word = first_word[0].lower() + first_word[1:]
                    s = f"{op}, {first_word} " + " ".join(words[1:])
                else:
                    prev_start = start_phrase
            else:
                prev_start = None
                
            rotated_sentences.append(s)
            
        sentences = rotated_sentences
        
        # 4. Remove Awkward Synonym Chains (Feature 3)
        synonym_adjectives = {
            "excellent", "splendid", "remarkable", "outstanding", "great", "good", 
            "high-quality", "favorable", "optimal", "effective", "efficient", 
            "reliable", "valuable", "exquisite", "stellar", "superb", "satisfactory", 
            "wonderful", "sound"
        }
        
        final_sentences = []
        for s in sentences:
            words = s.split()
            filtered_words = []
            skip_word = False
            
            for w_idx in range(len(words)):
                if skip_word:
                    skip_word = False
                    continue
                    
                w_clean = words[w_idx].lower().rstrip(",.")
                if w_clean in synonym_adjectives and w_idx < len(words) - 1:
                    w_next_clean = words[w_idx + 1].lower().rstrip(",.")
                    if w_next_clean in synonym_adjectives:
                        punc = ""
                        if words[w_idx].endswith(",") or words[w_idx].endswith("."):
                            punc = words[w_idx][-1]
                        
                        words[w_idx + 1] = words[w_idx + 1] + punc
                        continue
                        
                filtered_words.append(words[w_idx])
                
            final_sentences.append(" ".join(filtered_words))
            
        sentences = final_sentences
        
        # Join and clean up final double spaces
        res = " ".join(sentences).strip()
        res = re.sub(r"[^\S\r\n]{2,}", " ", res)
        res = re.sub(r"-\s+", "-", res)
        
        return res
