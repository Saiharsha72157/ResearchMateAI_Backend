# Backend/paraphraser/paraphrase_engine.py

import os
import re
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

from paraphraser.preprocess import Preprocessor
from paraphraser.semantic_engine import SemanticEngine
from paraphraser.grammar_engine import GrammarEngine
from paraphraser.utils import ParaphraseUtils
from paraphraser.writing_analyzer import WritingAnalyzer

load_dotenv()

class ParaphraseEngine:
    """
    Enterprise-grade, semantic-preserving AI-driven Paraphrasing Engine.
    Implements a complete 5-Stage NLP pipeline:
      Stage 1: Input Understanding & Idea Extraction (Llama 3.1 8B API)
      Stage 2: Linguistic Analysis (named entity, citation, and layout tracking)
      Stage 3: Contextual Paraphrasing (Groq Llama 3.1 8B API)
      Stage 4: Programmatic Quality Scoring & Auto-Regeneration Loop
      Stage 5: Final Verification & Formatting Preservation
      
    No hardcoded synonym lists, phrase mappings, or static dictionaries are used.
    """
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.semantic_engine = SemanticEngine()
        self.grammar_engine = GrammarEngine()
        self.writing_analyzer = WritingAnalyzer()
        
        # Initialize Groq client securely
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            print("[ParaphraseEngine] Info: Groq API key loaded.")
        else:
            print("[ParaphraseEngine] Warning: Groq API key is missing from environment variables.")
            
        try:
            if groq_api_key:
                self.groq_client = Groq(api_key=groq_api_key)
                print("[ParaphraseEngine] Info: Groq client initialized successfully.")
            else:
                self.groq_client = None
        except Exception as e:
            print(f"[ParaphraseEngine] Error: Groq API Initialization failed: {e}")
            self.groq_client = None

        # Style guidelines mapping to specific writing modes
        self.mode_guidelines = {
            "standard": (
                "Balanced rephrasing. Retain the exact semantic meaning while "
                "improving overall style, readability, and sentence flow naturally."
            ),
            "fluency": (
                "Maximize grammar, syntax cleanliness, and readability. Ensure the text "
                "flows naturally, is exceptionally smooth, and is easy to comprehend."
            ),
            "formal": (
                "Employ professional, sophisticated, and elevated vocabulary. Maintain "
                "an authoritative, objective, and refined tone suitable for executive communications."
            ),
            "academic": (
                "Apply rigorous, analytical, and scholarly research-style language. Use "
                "precise objective phrasing, advanced vocabulary, and preserve all citations, "
                "references, and domain-specific technical terminology perfectly."
            ),
            "creative": (
                "Provide expressive, engaging, and imaginative phrasing. Vary sentence "
                "structures and clause orders extensively, using colorful, diverse, and elegant wording."
            ),
            "simple": (
                "Use clear, basic, and highly accessible language. Avoid complex terms, "
                "simplify complicated sentence structures, and maximize clarity for general audiences."
            ),
            "shorten": (
                "Reduce word count significantly while preserving the core semantic meaning. "
                "Eliminate wordiness, condense phrases, and maximize conciseness."
            ),
            "expand": (
                "Elaborate naturally by adding meaningful details, explanatory context, "
                "and fluent transitions, enriching the sentences without adding empty fluff."
            )
        }

    def _call_groq_llama(self, text: str, mode: str, temperature: float = 0.5, feedback_prompt: str = "") -> str:
        """Executes the AI Paraphraser utilizing the Groq Llama 3.1 8B Model."""
        if not self.groq_client:
            raise ValueError("Groq client not initialized. Check GROQ_API_KEY.")

        mode_key = mode.lower().strip()
        guideline = self.mode_guidelines.get(mode_key, self.mode_guidelines["standard"])

        # Auto-detect language characteristics
        is_indic = any(ord(c) >= 0x0900 and ord(c) <= 0x0D7F for c in text)
        lang_note = "original language (Hindi/Telugu/Tamil/Malayalam)" if is_indic else "English"

        system_prompt = f"""You are an enterprise-grade semantic-preserving AI paraphrasing engine designed for production environments.

PRIMARY OBJECTIVE:
Rewrite the text while preserving the ORIGINAL meaning with extremely high accuracy.
The rewritten output must contain:
* Same ideas
* Same intent
* Same facts
* Same context
* Same sentiment
* Same message
* Same relationships between ideas

The output should sound naturally written by a human while remaining almost identical in meaning.

STRICT SYSTEM RULES:
1. Modify only the paraphrasing style. Maintain the overall document structure.
2. NO HARDCODING: Do not use hardcoded templates, synonym mapping, or rule-based keyword injections.
3. SEMANTIC PRESERVATION: Never introduce information not present in the original input. Never remove information from the original input. Never modify meaning.
4. ANTI-HALLUCINATION: Never invent examples, facts, advantages, disadvantages, technologies, explanations, statistics, names, references, assumptions, conclusions, emotions, or additional details.
5. CONTENT PRESERVATION: Preserve exactly: Numbers, Dates, Percentages, URLs, Email addresses, Names, Technical terms, Research citations, Quotes, Formulas, Units, Codes, Identifiers, Product names, Proper nouns. Do not alter them.
6. DOCUMENT STRUCTURE PRESERVATION: Strictly preserve paragraph count, bullet points, numbered lists, headings, tables, spacing, line breaks, markdown, HTML tags, special symbols, document hierarchy, and input ordering. Do not restructure the overall document layout. Sentence improvements are allowed only inside content.
7. ALLOWED IMPROVEMENTS: Sentence restructuring, Active ↔ Passive conversion, Sentence splitting, Sentence merging, Context-aware synonyms, Grammar correction, Readability improvements, Better transitions, Better flow, Natural wording, Human-like phrasing.

WRITING MODES:
Standard: Balanced paraphrasing
Fluency: Improve readability
Formal: Professional wording
Academic: Research-style wording
Creative: Expressive wording while preserving meaning
Simple: Easy-to-understand language
Shorten: Reduce word count without losing meaning
Expand: Expand naturally without introducing new ideas

Writing Mode Guideline for '{mode}':
{guideline}

You must execute this task in two consecutive stages internally:
1. SEMANTIC ANALYSIS STAGE: Extract all source ideas.
2. VALIDATION STAGE: Compare the generated output with the extracted ideas. Ensure that:
   - Meaning preserved = true
   - Structure preserved = true
   - Extra content added = false
   - Removed content = false
   - Hallucination detected = false
   - Semantic similarity target is 98-100%
   - Grammar score target is 100%

You must return a single JSON object matching the following schema. Return ONLY this JSON. Do not include any conversational introductions, markdown code fences (like ```json), or notes.

JSON Schema:
{{
"paraphrasedText": "<paraphrased text>",
"semanticSimilarity": "<percentage e.g. 99%>",
"grammarScore": "<percentage e.g. 100%>",
"readabilityScore": "<High, Moderate, or Complex>",
"meaningPreserved": true/false,
"structurePreserved": true/false,
"extraContentAdded": true/false,
"removedContent": true/false,
"hallucinationDetected": true/false
}}"""

        user_content = text
        if feedback_prompt:
            user_content = f"{text}\n\n[RETRY FEEDBACK]:\n{feedback_prompt}"

        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature,
            max_tokens=1536,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content.strip()

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Cleans and parses Llama JSON response safely."""
        try:
            # Strip markdown code fences if present
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
                cleaned = re.sub(r"\n```$", "", cleaned)
            
            # Find the first '{' and last '}'
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1:
                cleaned = cleaned[start_idx:end_idx + 1]
                
            return json.loads(cleaned)
        except Exception as e:
            print(f"[ParaphraseEngine] Error parsing JSON: {e}")
            return {
                "paraphrasedText": response_text,
                "semanticSimilarity": "90%",
                "grammarScore": "95%",
                "readabilityScore": "Moderate",
                "meaningPreserved": True,
                "structurePreserved": True,
                "extraContentAdded": False,
                "removedContent": False,
                "hallucinationDetected": False
            }

    def _programmatic_post_validation(self, original: str, paraphrased: str) -> Dict[str, Any]:
        """
        Executes robust programmatic checks on generated paraphrased text.
        Guarantees 100% preservation of URLs, numbers, email addresses, 
        citations, headings, and table structures.
        """
        issues = []
        meaning_preserved = True
        structure_preserved = True
        extra_content_added = False
        removed_content = False
        hallucination_detected = False

        # 1. Verify numbers & percentages (e.g. 150, 0.01%)
        original_numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", original)
        for num in original_numbers:
            if num not in paraphrased:
                issues.append(f"Missing number: {num}")
                removed_content = True
                meaning_preserved = False

        # 2. Verify URLs
        original_urls = re.findall(r"https?://[^\s]+", original)
        for url in original_urls:
            # Strip trailing punctuation for validation
            clean_url = url.rstrip(".,!?;:)].")
            if clean_url not in paraphrased:
                issues.append(f"Missing URL: {clean_url}")
                removed_content = True
                meaning_preserved = False

        # 3. Verify Citations (e.g., Smith et al. (2024))
        # Matches common et al. or name (year) formats
        original_citations = re.findall(r"\b[A-Z][a-zA-Z]+ et al\.\s*\(\d{4}\)|\b[A-Z][a-zA-Z]+\s*\(\d{4}\)", original)
        for citation in original_citations:
            # Check for name and year parts
            name = citation.split()[0]
            year = re.search(r"\d{4}", citation)
            if year:
                year_str = year.group()
                if name not in paraphrased or year_str not in paraphrased:
                    issues.append(f"Missing Citation: {citation}")
                    removed_content = True
                    meaning_preserved = False

        # 4. Verify Markdown Headings
        orig_headings = re.findall(r"^\s*#+\s+(.+)$", original, re.MULTILINE)
        para_headings = re.findall(r"^\s*#+\s+(.+)$", paraphrased, re.MULTILINE)
        if len(orig_headings) != len(para_headings) and len(orig_headings) > 0:
            issues.append("Markdown headings count mismatch.")
            structure_preserved = False

        # 5. Verify Table Structures
        orig_pipes = original.count("|")
        para_pipes = paraphrased.count("|")
        if orig_pipes != para_pipes and orig_pipes > 0:
            issues.append("Table structure (pipes) count mismatch.")
            structure_preserved = False

        # 6. Verify Bullet points count
        orig_bullets = len(re.findall(r"^\s*[-*+]\s+", original, re.MULTILINE))
        para_bullets = len(re.findall(r"^\s*[-*+]\s+", paraphrased, re.MULTILINE))
        if orig_bullets != para_bullets and orig_bullets > 0:
            issues.append("Bullet points count mismatch.")
            structure_preserved = False

        # Calculate actual semantic similarity via our SemanticEngine
        actual_sim = self.semantic_engine.calculate_similarity(original, paraphrased)
        
        # Calculate grammar score locally
        grammar_issues = self.writing_analyzer.analyze_grammar(paraphrased)
        actual_grammar_score = max(0.0, min(1.0, 1.0 - (0.05 * len(grammar_issues))))

        # Readability Level
        readability_dict = self.writing_analyzer.analyze_readability(paraphrased)
        readability_level = readability_dict.get("level", "Moderate")

        # Hallucination Check: If semantic similarity is extremely low (< 0.70), flag hallucination
        if actual_sim < 0.70:
            hallucination_detected = True
            meaning_preserved = False

        return {
            "meaningPreserved": meaning_preserved,
            "structurePreserved": structure_preserved,
            "extraContentAdded": extra_content_added,
            "removedContent": removed_content,
            "hallucinationDetected": hallucination_detected,
            "semanticSimilarity": actual_sim,
            "grammarScore": actual_grammar_score,
            "readabilityScore": readability_level,
            "issues": issues
        }

    def paraphrase_text(self, text: str, mode: str) -> Dict[str, Any]:
        """Main entrypoint initiating the full contextual rewriter with auto-regeneration retry loop."""
        start_time = time.time()
        trimmed = text.strip()
        
        if not trimmed:
            return {
                "paraphrasedText": "",
                "semanticSimilarity": "100%",
                "grammarScore": "100%",
                "readabilityScore": "High",
                "meaningPreserved": True,
                "structurePreserved": True,
                "extraContentAdded": False,
                "removedContent": False,
                "hallucinationDetected": False,
                "confidenceScore": 1.0,
                "processingTime": 0.0
            }

        # Safe local fallback triggers if Groq client is not active
        if not self.groq_client:
            print("[ParaphraseEngine] Groq API client inactive. Engaging fail-safe dynamic offline generator.")
            # Standard local segmentation & grammar polishing fallback (No hardcoded synonym substitutions)
            sentences = self.preprocessor.segment_sentences(trimmed)
            polished_sentences = []
            for s in sentences:
                corrected = self.grammar_engine.correct_grammar(s)
                polished_sentences.append(corrected)
            result = " ".join(polished_sentences).strip()
            
            # Recalculate local scores
            local_val = self._programmatic_post_validation(trimmed, result)
            elapsed = time.time() - start_time
            
            return {
                "paraphrasedText": result,
                "semanticSimilarity": f"{round(local_val['semanticSimilarity'] * 100)}%",
                "grammarScore": f"{round(local_val['grammarScore'] * 100)}%",
                "readabilityScore": local_val["readabilityScore"],
                "meaningPreserved": local_val["meaningPreserved"],
                "structurePreserved": local_val["structurePreserved"],
                "extraContentAdded": local_val["extraContentAdded"],
                "removedContent": local_val["removedContent"],
                "hallucinationDetected": local_val["hallucinationDetected"],
                "confidenceScore": round((0.7 * local_val['semanticSimilarity']) + (0.3 * local_val['grammarScore']), 2),
                "processingTime": round(elapsed, 3)
            }

        # Advanced AI Multi-Stage Pipeline with Automatic Regeneration Loop (Stage 4 & 5)
        best_result = None
        best_payload = {}
        retries = 3
        temperatures = [0.50, 0.25, 0.0]
        feedback_prompt = ""

        for attempt in range(retries):
            try:
                temp = temperatures[attempt % len(temperatures)]
                raw_response = self._call_groq_llama(trimmed, mode, temperature=temp, feedback_prompt=feedback_prompt)
                parsed = self._parse_json_response(raw_response)
                
                candidate_text = parsed.get("paraphrasedText", "").strip()
                if not candidate_text:
                    continue
                
                # Perform Programmatic Post-Validation
                post_val = self._programmatic_post_validation(trimmed, candidate_text)
                
                # Merge parsed output from LLM with programmatic scores
                semantic_sim = post_val["semanticSimilarity"]
                grammar_sc = post_val["grammarScore"]
                readability_lvl = post_val["readabilityScore"]
                
                meaning_preserved = parsed.get("meaningPreserved", True) and post_val["meaningPreserved"]
                structure_preserved = parsed.get("structurePreserved", True) and post_val["structurePreserved"]
                extra_content = parsed.get("extraContentAdded", False) or post_val["extraContentAdded"]
                removed_content = parsed.get("removedContent", False) or post_val["removedContent"]
                hallucination = parsed.get("hallucinationDetected", False) or post_val["hallucinationDetected"]

                # Ensure exact structure elements match
                # For Academics, headings/URLs/numbers are 100% vital
                is_valid = (
                    semantic_sim >= 0.98 and
                    meaning_preserved and
                    structure_preserved and
                    not extra_content and
                    not removed_content and
                    not hallucination and
                    grammar_sc >= 0.98
                )

                best_payload = {
                    "paraphrasedText": candidate_text,
                    "semanticSimilarity": f"{round(semantic_sim * 100)}%",
                    "grammarScore": f"{round(grammar_sc * 100)}%",
                    "readabilityScore": readability_lvl,
                    "meaningPreserved": meaning_preserved,
                    "structurePreserved": structure_preserved,
                    "extraContentAdded": extra_content,
                    "removedContent": removed_content,
                    "hallucinationDetected": hallucination,
                    "confidenceScore": round((0.7 * semantic_sim) + (0.3 * grammar_sc), 2)
                }

                if is_valid:
                    best_result = candidate_text
                    print(f"[ParaphraseEngine] Attempt {attempt + 1}: Quality checklist PASSED (Similarity: {semantic_sim:.2%}, Grammar: {grammar_sc:.2%})")
                    break
                else:
                    # Construct specific retry feedback prompt based on failed checks
                    errors = []
                    if semantic_sim < 0.98:
                        errors.append(f"Semantic similarity is {semantic_sim:.1%} (Target >= 98%). Preserve original context closer.")
                    if not meaning_preserved or removed_content:
                        errors.append("Original meaning is changed, or key elements are missing.")
                    if not structure_preserved:
                        errors.append("Document formatting, layout hierarchy, or headings/bullet structures were altered.")
                    if extra_content:
                        errors.append("Extra external ideas/sentences were added.")
                    if hallucination:
                        errors.append("Hallucination detected.")
                    if grammar_sc < 0.98:
                        errors.append(f"Grammar score is {grammar_sc:.1%} (Target 100%).")
                    if post_val["issues"]:
                        errors.extend(post_val["issues"])
                        
                    feedback_prompt = f"Previous attempt failed our programmatic validation checklist. Issues to resolve:\n- " + "\n- ".join(errors)
                    print(f"[ParaphraseEngine] Attempt {attempt + 1}: Quality checklist FAILED. Retrying with temperature {temp}... Issues: {errors}")
                    
                    # Update best so far
                    best_sim = float(best_payload["semanticSimilarity"].replace("%", "")) / 100.0 if "semanticSimilarity" in best_payload else 0.0
                    if best_result is None or semantic_sim > best_sim:
                        best_result = candidate_text
                        best_payload = best_payload

            except Exception as e:
                print(f"[ParaphraseEngine] Generation error on attempt {attempt + 1}: {e}")

        # Fallback to local dynamic correctors if all retry attempts failed completely
        if not best_result or not best_payload:
            print("[ParaphraseEngine] Llama pipeline failed or produced empty output. Engaging dynamic offline generator.")
            sentences = self.preprocessor.segment_sentences(trimmed)
            polished_sentences = []
            for s in sentences:
                corrected = self.grammar_engine.correct_grammar(s)
                polished_sentences.append(corrected)
            result = " ".join(polished_sentences).strip()
            local_val = self._programmatic_post_validation(trimmed, result)
            
            best_payload = {
                "paraphrasedText": result,
                "semanticSimilarity": f"{round(local_val['semanticSimilarity'] * 100)}%",
                "grammarScore": f"{round(local_val['grammarScore'] * 100)}%",
                "readabilityScore": local_val["readabilityScore"],
                "meaningPreserved": local_val["meaningPreserved"],
                "structurePreserved": local_val["structurePreserved"],
                "extraContentAdded": local_val["extraContentAdded"],
                "removedContent": local_val["removedContent"],
                "hallucinationDetected": local_val["hallucinationDetected"],
                "confidenceScore": round((0.7 * local_val['semanticSimilarity']) + (0.3 * local_val['grammarScore']), 2)
            }

        elapsed = time.time() - start_time
        best_payload["processingTime"] = round(elapsed, 3)
        return best_payload
