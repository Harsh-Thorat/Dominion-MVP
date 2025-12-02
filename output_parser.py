"""
Custom output parser for exercise feedback
"""
import re
from typing import Dict, List, Any
from langchain.schema import BaseOutputParser


class ExerciseFeedbackParser(BaseOutputParser):
    """
    Parse raw LLM output into structured exercise feedback
    """
    
    # Keywords for body parts
    BODY_PARTS = [
        "knees", "knee", "back", "spine", "shoulders", "shoulder",
        "elbows", "elbow", "hips", "hip", "core", "chest",
        "arms", "arm", "legs", "leg", "head", "neck", "wrists", "wrist"
    ]
    
    # Form rating keywords
    RATINGS = {
        "good": ["good form", "excellent", "great", "perfect", "correct"],
        "minor": ["minor issue", "slight", "small issue", "needs adjustment"],
        "major": ["major issue", "incorrect", "dangerous", "wrong", "poor"]
    }
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse LLM output into structured feedback
        
        Args:
            text: Raw text from LLM
        
        Returns:
            Dictionary with structured feedback
        """
        # Extract form rating
        rating = self._extract_rating(text)
        
        # Extract mentioned body parts
        body_parts = self._extract_body_parts(text)
        
        # Clean feedback text
        feedback = self._clean_feedback(text)
        
        return {
            "rating": rating,
            "feedback": feedback,
            "body_parts": body_parts,
            "raw_output": text
        }
    
    def _extract_rating(self, text: str) -> str:
        """
        Extract form rating from text
        """
        text_lower = text.lower()
        
        # Check for explicit rating mentions
        for rating, keywords in self.RATINGS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if rating == "good":
                        return "Good"
                    elif rating == "minor":
                        return "Minor Issue"
                    else:
                        return "Major Issue"
        
        # Default to neutral if no clear rating found
        return "Unable to Rate"
    
    def _extract_body_parts(self, text: str) -> List[str]:
        """
        Extract mentioned body parts from text
        """
        text_lower = text.lower()
        found_parts = []
        
        for part in self.BODY_PARTS:
            if part in text_lower and part not in found_parts:
                # Add base form (remove 's' from plural)
                base_part = part.rstrip('s')
                if base_part not in found_parts:
                    found_parts.append(part)
        
        return found_parts
    
    def _clean_feedback(self, text: str) -> str:
        """
        Clean and format feedback text
        """
        # Remove common prefixes
        text = re.sub(r'^(feedback:|analysis:|rating:)\s*', '', text, flags=re.IGNORECASE)
        
        # Remove explicit rating mentions at the end
        text = re.sub(r'\s*(form rating|rating):\s*\w+[\w\s]*$', '', text, flags=re.IGNORECASE)
        
        # Clean whitespace
        text = ' '.join(text.split())
        
        # Ensure it ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text.strip()
    
    @property
    def _type(self) -> str:
        return "exercise_feedback"
    
    def get_format_instructions(self) -> str:
        """
        Return format instructions for the LLM
        """
        return """Provide feedback in 1-3 clear sentences. Focus on the most important form issues.
End with a form rating using one of these exact phrases:
- "Good form" (if technique is correct)
- "Minor issue" (if there's a small problem)
- "Major issue" (if there's a serious form problem)

Example: "Your squat depth is good, but your knees are caving inward slightly. Keep them aligned with your toes. Minor issue."
"""
