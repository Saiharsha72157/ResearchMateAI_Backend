# Backend/paraphraser/modes.py

class WritingModes:
    """
    Houses prompt configurations, T5 instruction prefixes, and weighting parameters 
    for the 8 supported writing modes.
    """
    MODES = {
        "standard": {
            "label": "Standard",
            "t5_prefix": "paraphrase: ",
            "semantic_weight": 0.40,
            "grammar_weight": 0.30,
            "readability_weight": 0.20,
            "diversity_weight": 0.10
        },
        "fluency": {
            "label": "Fluency",
            "t5_prefix": "fluent paraphrase: ",
            "semantic_weight": 0.35,
            "grammar_weight": 0.30,
            "readability_weight": 0.25, # High readability priority
            "diversity_weight": 0.10
        },
        "formal": {
            "label": "Formal",
            "t5_prefix": "formal paraphrase: ",
            "semantic_weight": 0.40,
            "grammar_weight": 0.35, # High grammar expectation
            "readability_weight": 0.15,
            "diversity_weight": 0.10
        },
        "academic": {
            "label": "Academic",
            "t5_prefix": "academic paraphrase: ",
            "semantic_weight": 0.45, # Strict semantic preservation
            "grammar_weight": 0.35,
            "readability_weight": 0.10, # Readability can be lower/more complex
            "diversity_weight": 0.10
        },
        "creative": {
            "label": "Creative",
            "t5_prefix": "creative paraphrase: ",
            "semantic_weight": 0.30,
            "grammar_weight": 0.25,
            "readability_weight": 0.20,
            "diversity_weight": 0.25 # High priority for vocabulary variety
        },
        "expand": {
            "label": "Expand",
            "t5_prefix": "expand paraphrase: ",
            "semantic_weight": 0.40,
            "grammar_weight": 0.30,
            "readability_weight": 0.15,
            "diversity_weight": 0.15
        },
        "shorten": {
            "label": "Shorten",
            "t5_prefix": "shorten paraphrase: ",
            "semantic_weight": 0.40,
            "grammar_weight": 0.30,
            "readability_weight": 0.20,
            "diversity_weight": 0.10
        },
        "simple": {
            "label": "Simple",
            "t5_prefix": "simple paraphrase: ",
            "semantic_weight": 0.35,
            "grammar_weight": 0.30,
            "readability_weight": 0.25, # Simple vocabulary matches easy readability
            "diversity_weight": 0.10
        }
    }

    @staticmethod
    def get_prefix(mode: str) -> str:
        """Retrieves T5 prompt prefix for the selected mode."""
        mode_key = mode.lower().strip()
        return WritingModes.MODES.get(mode_key, WritingModes.MODES["standard"])["t5_prefix"]

    @staticmethod
    def get_weights(mode: str) -> dict:
        """Retrieves score ranker weights for the selected mode."""
        mode_key = mode.lower().strip()
        data = WritingModes.MODES.get(mode_key, WritingModes.MODES["standard"])
        return {
            "semantic": data["semantic_weight"],
            "grammar": data["grammar_weight"],
            "readability": data["readability_weight"],
            "diversity": data["diversity_weight"]
        }
