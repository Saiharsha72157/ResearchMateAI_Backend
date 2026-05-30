# Backend/routes/paraphrase.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from paraphraser import CandidateRanker
from paraphraser.history_manager import HistoryManager
from paraphraser.writing_analyzer import WritingAnalyzer
from paraphraser.humanizer import Humanizer

router = APIRouter()
ranker = CandidateRanker()
history_manager = HistoryManager()

import os
import re
import json
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client locally to avoid circular imports
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"[Backend] Groq API Initialization Warning in paraphrase router: {e}")
    groq_client = None


class ParaphraseRequest(BaseModel):
    text: str
    mode: str = "standard"
    language: Optional[str] = "English"

@router.post("/paraphrase")
def paraphrase_text(data: ParaphraseRequest):
    # Trim and validate input text
    text = data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
    
    # Word count validation of input
    words = text.split()
    if len(words) > 200:
        raise HTTPException(
            status_code=400, 
            detail=f"Text exceeds the 200-word limit ({len(words)} words). Please shorten your text."
        )
    
    # Mode validation for all 8 supported modes
    mode = data.mode.lower().strip()
    valid_modes = ["standard", "fluency", "formal", "academic", "creative", "expand", "shorten", "simple"]
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid paraphrasing mode '{data.mode}'. Supported modes are: {', '.join(valid_modes)}."
        )
        
    language = data.language or "English"
    language = language.strip().capitalize()
    
    try:
        # Automatically detect Indic script characters (Unicode range for Hindi, Telugu, Tamil, Malayalam)
        has_indic = any(ord(c) >= 0x0900 and ord(c) <= 0x0D7F for c in text)
        
        # Setup defaults for the enterprise metrics
        semantic_similarity = "95%"
        grammar_score = "100%"
        readability_score = "High"
        meaning_preserved = True
        structure_preserved = True
        extra_content_added = False
        removed_content = False
        hallucination_detected = False
        
        if (language != "English" or has_indic) and groq_client is not None:
            target_lang = language
            if has_indic and language == "English":
                target_lang = "native Indic script language (Hindi/Telugu/Tamil/Malayalam)"
                
            print(f"[Backend] Bilingual AI Paraphraser engaged. Language: {target_lang}, Mode: {mode}")
            
            system_prompt = f"""You are an enterprise-grade bilingual semantic-preserving AI paraphrasing and translation engine designed for production environments.

PRIMARY OBJECTIVE:
Translate and paraphrase the text to the target language '{target_lang}' while preserving the ORIGINAL meaning with extremely high accuracy.
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
1. Translate/Paraphrase the text to the target language '{target_lang}' matching the mode '{mode}'.
2. NO HARDCODING: Do not use hardcoded templates or synonym lists.
3. SEMANTIC PRESERVATION: Never introduce information not present in the original input. Never remove information.
4. ANTI-HALLUCINATION: Never invent facts, explanations, statistics, or details.
5. CONTENT PRESERVATION: Preserve exactly: Numbers, Dates, Percentages, URLs, Email addresses, Names, Technical terms, Research citations, Quotes, Formulas, Units. Do not alter them.
6. DOCUMENT STRUCTURE PRESERVATION: Strictly preserve paragraph count, headings, tables, bullet points, spacing, and line breaks exactly as provided.
7. Return ONLY the JSON object matching the schema below. No conversational text or markdown fences.

JSON Schema:
{{
"paraphrasedText": "<translated and paraphrased text in target language>",
"semanticSimilarity": "<percentage e.g. 98%>",
"grammarScore": "<percentage e.g. 100%>",
"readabilityScore": "<High, Moderate, or Complex>",
"meaningPreserved": true/false,
"structurePreserved": true/false,
"extraContentAdded": true/false,
"removedContent": true/false,
"hallucinationDetected": true/false
}}"""

            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            result_raw = response.choices[0].message.content.strip()
            
            # Clean and parse JSON safely
            cleaned = result_raw
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
                cleaned = re.sub(r"\n```$", "", cleaned)
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1:
                cleaned = cleaned[start_idx:end_idx + 1]
            
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = {}
                
            result = parsed.get("paraphrasedText", result_raw).strip()
            if result.startswith('"') and result.endswith('"') and len(result) > 2:
                result = result[1:-1].strip()
                
            # Perform post validation metrics programmatically for bilingual
            post_val = ranker.generator._programmatic_post_validation(text, result)
            
            semantic_similarity = parsed.get("semanticSimilarity", f"{round(post_val['semanticSimilarity']*100)}%")
            grammar_score = parsed.get("grammarScore", f"{round(post_val['grammarScore']*100)}%")
            readability_score = parsed.get("readabilityScore", post_val["readabilityScore"])
            meaning_preserved = parsed.get("meaningPreserved", True) and post_val["meaningPreserved"]
            structure_preserved = parsed.get("structurePreserved", True) and post_val["structurePreserved"]
            extra_content_added = parsed.get("extraContentAdded", False) or post_val["extraContentAdded"]
            removed_content = parsed.get("removedContent", False) or post_val["removedContent"]
            hallucination_detected = parsed.get("hallucinationDetected", False) or post_val["hallucinationDetected"]
            
            # Extract float for compatibility score
            sim_str = semantic_similarity.replace("%", "").strip()
            try:
                sim_val = float(sim_str) / 100.0 if "%" in semantic_similarity else float(sim_str)
            except Exception:
                sim_val = 0.90
            score = sim_val * 10.0
            
        else:
            # Generate candidates, evaluate them, and select the highest scoring output
            clean_text = ranker.preprocessor.sanitize_text(text)
            res_dict = ranker.generator.paraphrase_text(clean_text, mode=mode)
            result = Humanizer.humanize(res_dict["paraphrasedText"])
            
            semantic_similarity = res_dict["semanticSimilarity"]
            grammar_score = res_dict["grammarScore"]
            readability_score = res_dict["readabilityScore"]
            meaning_preserved = res_dict["meaningPreserved"]
            structure_preserved = res_dict["structurePreserved"]
            extra_content_added = res_dict["extraContentAdded"]
            removed_content = res_dict["removedContent"]
            hallucination_detected = res_dict["hallucinationDetected"]
            score = res_dict.get("confidenceScore", 0.90) * 10.0

        # Save record to persistent local history log
        history_manager.save_history(
            original_text=text,
            paraphrased_text=result,
            mode=mode,
            score=score
        )
        
        # Calculate statistics
        result_words = result.split()
        word_count = len(result_words)
        
        # Safe sentence count calculation
        sentence_count = max(1, len(ranker.preprocessor.segment_sentences(result)))
        
        return {
            # Old backward-compatible keys
            "original_text": text,
            "paraphrased_text": result,
            "mode": mode,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "score": round(score, 1),
            
            # New enterprise keys
            "paraphrasedText": result,
            "semanticSimilarity": semantic_similarity,
            "grammarScore": grammar_score,
            "readabilityScore": readability_score,
            "meaningPreserved": meaning_preserved,
            "structurePreserved": structure_preserved,
            "extraContentAdded": extra_content_added,
            "removedContent": removed_content,
            "hallucinationDetected": hallucination_detected
        }
    except Exception as e:
        print(f"[Backend] Local Paraphrasing error: {e}")
        raise HTTPException(status_code=500, detail=f"Paraphrasing failed: {str(e)}")

@router.get("/paraphrase/history")
def get_paraphrase_history():
    """Retrieves all local persistent history records."""
    try:
        return history_manager.get_history()
    except Exception as e:
        print(f"[Backend] Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@router.get("/paraphrase/favorites")
def get_paraphrase_favorites():
    """Retrieves all starred favorite rephrasing records."""
    try:
        return history_manager.get_favorites()
    except Exception as e:
        print(f"[Backend] Error fetching favorites: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch favorites: {str(e)}")

@router.post("/paraphrase/favorite/{entry_id}")
def toggle_paraphrase_favorite(entry_id: str):
    """Toggles the favorite state of a history entry."""
    try:
        updated = history_manager.toggle_favorite(entry_id)
        if not updated:
            raise HTTPException(status_code=404, detail=f"History record with ID {entry_id} not found.")
        return updated
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[Backend] Error toggling favorite: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to toggle favorite: {str(e)}")

@router.delete("/paraphrase/history/{entry_id}")
def delete_paraphrase_history(entry_id: str):
    """Deletes a history record by unique identifier."""
    try:
        deleted = history_manager.delete_history(entry_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"History record with ID {entry_id} not found.")
        return {"success": True, "message": "History entry deleted successfully."}
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[Backend] Error deleting history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete history: {str(e)}")

# Initialize writing intelligence analyzer
writing_analyzer = WritingAnalyzer()

class WritingAnalysisRequest(BaseModel):
    text: str

@router.post("/writing-analysis")
def analyze_writing_text(data: WritingAnalysisRequest):
    """Runs granular grammar checking, readability indexing, tone detection, and suggestions."""
    text = data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
    
    try:
        grammar = writing_analyzer.analyze_grammar(text)
        readability = writing_analyzer.analyze_readability(text)
        tone = writing_analyzer.detect_tone(text)
        suggestions = writing_analyzer.generate_suggestions(text)
        
        return {
            "grammar": grammar,
            "readability": readability,
            "tone": tone,
            "suggestions": suggestions
        }
    except Exception as e:
        print(f"[Backend] Writing analysis endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Writing analysis failed: {str(e)}")
