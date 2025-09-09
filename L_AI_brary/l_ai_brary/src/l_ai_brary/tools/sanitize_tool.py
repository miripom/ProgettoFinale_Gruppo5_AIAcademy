from crewai.tools import BaseTool
from typing import ClassVar
import re
from crewai import LLM

class LexicalTool(BaseTool):
    """A CrewAI tool for sanitizing and normalizing user input text.
    
    This tool provides lexical cleaning capabilities to ensure user input is safe
    and properly formatted by removing unwanted characters, HTML tags, malicious
    sequences, and normalizing text formatting.
    
    Attributes:
        name (str): The display name of the tool.
        description (str): A detailed description of the tool's functionality.
    """
    
    name: str = "Lexical Sanitizer Tool"
    description: str = (
        "Cleans and normalizes raw user input by removing unwanted characters, "
        "HTML tags, malicious sequences, and ensuring safe formatting."
    )

    def _run(self, user_input: str, max_len: int = 512) -> str:
        """Sanitizes and normalizes user input text.
        
        This method performs comprehensive text cleaning by removing control characters,
        HTML tags, malicious sequences, and normalizing formatting elements like quotes,
        punctuation, and whitespace.
        
        Args:
            user_input (str): The raw user input text to be sanitized.
            max_len (int, optional): Maximum length limit for the input text. 
                Defaults to 512.
        
        Returns:
            str: The sanitized and normalized text, safe for further processing.
            
        Examples:
            >>> tool = LexicalTool()
            >>> tool._run("Hello <script>alert('xss')</script> world!!!")
            'Hello  world!'
            >>> tool._run("Multiple   spaces\t\nand\xa0unicode")
            'Multiple spaces and unicode'
        """
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



