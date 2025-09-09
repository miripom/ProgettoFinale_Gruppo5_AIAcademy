"""Image Generation Tool for CrewAI.

This module provides a CrewAI tool for generating images using Azure OpenAI's DALL-E model.
The tool takes text descriptions and generates corresponding images, saving them to the
specified output directory.

Classes:
    GenerateImageInput: Pydantic model defining the input schema for image generation.
    ImageGenerationTool: CrewAI tool that handles the image generation workflow.

Dependencies:
    - Azure OpenAI API access with image generation capabilities
    - Environment variables for API configuration
    - Base64 encoding/decoding for image data handling
"""

import os
import base64
import time
from typing import Any, Type, List
from pydantic import Field
from dotenv import load_dotenv
from openai import AzureOpenAI, BaseModel
from crewai.tools import BaseTool


class GenerateImageInput(BaseModel):
	"""Input schema for the ImageGenerationTool.

	This Pydantic model defines the required input parameters for generating
	images using the Azure OpenAI DALL-E model through the CrewAI framework.

	Attributes:
		prompt (str): Detailed description of the image to generate. Should be
			descriptive and specific to get better results from the AI model.
		path (str): Directory path where the generated image should be saved.
			The directory will be created if it doesn't exist.
		security_context (Any): Security context information for the image
			generation process. Used for authentication and authorization.
	"""
	prompt: str = Field(..., description="Description of the image to generate.")
	path: str = Field(..., description="Path to the folder where image should be saved.")
	security_context: Any = Field(..., description="Security context for the image generation.")

class ImageGenerationTool(BaseTool):
	"""CrewAI tool for generating images using Azure OpenAI DALL-E model.
	
	This tool integrates with Azure OpenAI's image generation service to create
	images based on text descriptions. Generated images are saved as PNG files
	with timestamps to ensure unique filenames.
	
	The tool handles the complete workflow:
	- Loading environment variables for API configuration
	- Connecting to Azure OpenAI service
	- Generating images from text prompts
	- Decoding base64 image data
	- Saving images to the specified directory
	
	Attributes:
		name (str): Display name of the tool for CrewAI.
		description (str): Description of the tool's functionality.
		args_schema (Type[BaseModel]): Pydantic schema for input validation.
		
	Environment Variables Required:
		AZURE_API_KEY: Azure OpenAI API key
		AZURE_OPENAI_API_VERSION: API version to use
		AZURE_API_BASE: Azure OpenAI endpoint URL
		DEPLOYMENT_IMAGE_GENERATION: Name of the image generation deployment
	"""

	name: str = "Image Generation Tool"
	description: str = (
		"A tool to generate images based on a description."
	)
	args_schema: Type[BaseModel] = GenerateImageInput

	def _run(self, prompt: str, path: str, security_context) -> str:
		"""Execute the image generation process.
		
		This method handles the complete image generation workflow:
		1. Loads environment variables for Azure OpenAI configuration
		2. Establishes connection to Azure OpenAI service
		3. Generates an image based on the provided text prompt
		4. Decodes the base64-encoded image data
		5. Saves the image to the specified directory with a timestamp filename
		
		Args:
			prompt (str): Descriptive text for the image to generate. More detailed
				prompts typically yield better results.
			path (str): Directory path where the image should be saved. Will be
				created if it doesn't exist.
			security_context: Security context for the operation (currently unused
				but required by the interface).
				
		Returns:
			str: Full file path of the generated image file, or error information
				if the generation failed.
				
		Raises:
			ValueError: If the API returns no base64 image data.
			Exception: For various errors during image generation or file operations.
			
		Note:
			The generated image is always 1024x1024 pixels in PNG format.
			The filename includes a timestamp to ensure uniqueness.
		"""

		load_dotenv()

		# Initialize Azure OpenAI client with environment variables
		client = AzureOpenAI(
				api_key=os.getenv("AZURE_API_KEY") or "",
				api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "",
				azure_endpoint=os.getenv("AZURE_API_BASE") or "",
		)

		# Generate image using DALL-E model
		result = client.images.generate(
				model=os.getenv("DEPLOYMENT_IMAGE_GENERATION"),
				prompt=prompt,
				size="1024x1024",   # options: 256x256, 512x512, 1024x1024
				response_format="b64_json",
		)
		
		# Override path to default output directory
		path = 'output'
		
		try:
			# Validate and process the API response
			if result and result.data and len(result.data) > 0:
				image_base64 = result.data[0].b64_json
				if image_base64 is None:
					raise ValueError("No base64 image returned by the API")
				
				# Decode base64 image data
				image_bytes = base64.b64decode(image_base64)

				# Create output directory if it doesn't exist
				os.makedirs(path, exist_ok=True)

				# Generate unique filename with timestamp
				filename = os.path.join(path, f"generated_{int(time.time())}.png")
				
				# Save image to file
				with open(filename, "wb") as f:
					f.write(image_bytes)

		except Exception as e:
			print(f"Error generating image: {e}")
			return f"Error generating image: {e}"

		return filename