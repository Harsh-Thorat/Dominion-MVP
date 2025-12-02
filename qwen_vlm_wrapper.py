"""
Custom LangChain LLM wrapper for Qwen2.5-VL
"""
from typing import Any, List, Optional, Dict
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


class Qwen2VLWrapper(LLM):
    """
    LangChain wrapper for Qwen2.5-VL vision-language model
    """
    
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    model: Any = None
    processor: Any = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 256
    temperature: float = 0.7
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen2.5-VL model and processor"""
        print(f"Loading {self.model_name} on {self.device}...")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        print("Model loaded successfully!")
    
    @property
    def _llm_type(self) -> str:
        return "qwen2vl"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute the model with text prompt only
        This is called by LangChain's standard interface
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        return self._generate(messages, stop, run_manager, **kwargs)
    
    def call_with_frames(
        self,
        prompt: str,
        frames: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute the model with both text prompt and video frames
        
        Args:
            prompt: Text instruction
            frames: List of PIL Images
            stop: Stop sequences
            run_manager: Callback manager
        
        Returns:
            Generated text response
        """
        # Construct multimodal message
        content = []
        
        # Add video frames
        for frame in frames:
            content.append({
                "type": "image",
                "image": frame
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return self._generate(messages, stop, run_manager, **kwargs)
    
    def _generate(
        self,
        messages: List[Dict],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Internal method to generate response from messages
        """
        # Prepare inputs using Qwen's processor
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Tokenize
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
            )
        
        # Decode output
        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):] 
            for i in range(len(output_ids))
        ]
        
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Handle streaming callbacks
        if run_manager:
            for token in response.split():
                run_manager.on_llm_new_token(token + " ")
        
        return response
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters"""
        return {
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
