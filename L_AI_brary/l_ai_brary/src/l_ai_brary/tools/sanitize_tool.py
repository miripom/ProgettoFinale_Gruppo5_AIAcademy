from crewai.tools import BaseTool
from typing import ClassVar
import re
from crewai import LLM

class LexicalTool(BaseTool):
    name: str = "Lexical Sanitizer Tool"
    description: str = (
        "Cleans and normalizes raw user input by removing unwanted characters, "
        "HTML tags, malicious sequences, and ensuring safe formatting."
    )

    def _run(self, user_input: str, max_len: int = 512) -> str:
        if not user_input:
            return ""

        # Trim and limit length
        safe = user_input.strip()
        safe = safe[:max_len]

        # Remove control characters
        safe = re.sub(r"[\x00-\x1F\x7F]", "", safe)
        
        # Remove HTML/markup
        safe = re.sub(r"<.*?>", "", safe)

        # Normalize Unicode spaces, tabs, line breaks
        safe = re.sub(r"[\u200b-\u200f\u202a-\u202e\t\n\r\xa0]+", " ", safe)

        # Normalize quotes and apostrophes
        safe = safe.replace("“", '"').replace("”", '"').replace("’", "'")

        # Normalize multiple punctuation marks (!!! -> !)
        safe = re.sub(r"([!?,]){2,}", r"\1", safe)

        # Remove multiple spaces
        safe = re.sub(r"\s+", " ", safe)

        return safe



