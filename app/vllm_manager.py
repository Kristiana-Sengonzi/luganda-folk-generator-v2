# app/vllm_manager.py - LLM MODELS ONLY
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from app.config import STORY_GENERATOR_PATH, LYRIC_GENERATOR_PATH,BASE_MODEL_PATH

class VLLMManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm_engine = None

    def initialize_llm(self):
        """Initialize vLLM engine with LoRA support - LLM MODELS ONLY"""
        if self.llm_engine is None:
            print("Initializing vLLM engine for LLM models...")
            self.llm_engine = LLM(
                model=BASE_MODEL_PATH,  
                tokenizer=BASE_MODEL_PATH,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                max_model_len=4096,
                enable_lora=True,
                max_lora_rank=64,
                max_loras=4
            )
           
        return self.llm_engine
        

    def generate_story(self, prompt, max_tokens=300):
        """Generate story using story LoRA adapter"""
        self.initialize_llm()
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=max_tokens,
            repetition_penalty=1.8,
            stop_token_ids=[2]
        )
        
        outputs = self.llm_engine.generate(
            [prompt],
            sampling_params,
            lora_request=LoRARequest(STORY_GENERATOR_PATH, 1)  # Story LoRA
        )
        
        return outputs[0].outputs[0].text

    def generate_lyrics(self, prompt, max_tokens=512):
        """Generate lyrics using lyric LoRA adapter"""
        self.initialize_llm()
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=max_tokens,
            repetition_penalty=1.2,
            stop_token_ids=[2]
        )
        
        outputs = self.llm_engine.generate(
            [prompt],
            sampling_params,
            lora_request=LoRARequest(LYRIC_GENERATOR_PATH, 2)  # Lyric LoRA
        )
        
        return outputs[0].outputs[0].text

# Singleton instance
vllm_manager = VLLMManager()
