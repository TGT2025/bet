"""
🤖 Model Factory - Multi-LLM Interface
Unified interface for calling different LLM providers
"""

import os
from typing import Optional, Dict, Any
from src import config

class ModelFactory:
    """Factory for creating and calling different LLM models"""
    
    @staticmethod
    def call_llm(
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Call an LLM with the given prompt
        
        Args:
            model: Model name (e.g., "gpt-4", "claude-3-5-sonnet", "deepseek-reasoner")
            prompt: The user prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            The model's response as a string
        """
        
        # Route to appropriate provider
        if model.startswith("gpt-") or model.startswith("o1-"):
            return ModelFactory._call_openai(model, prompt, temperature, max_tokens, system_prompt)
        elif model.startswith("claude-"):
            return ModelFactory._call_anthropic(model, prompt, temperature, max_tokens, system_prompt)
        elif model.startswith("deepseek-"):
            return ModelFactory._call_deepseek(model, prompt, temperature, max_tokens, system_prompt)
        elif model.startswith("gemini-"):
            return ModelFactory._call_google(model, prompt, temperature, max_tokens, system_prompt)
        else:
            raise ValueError(f"Unknown model type: {model}")
    
    @staticmethod
    def _call_openai(model: str, prompt: str, temperature: float, max_tokens: int, system_prompt: Optional[str]) -> str:
        """Call OpenAI API"""
        import openai
        
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    @staticmethod
    def _call_anthropic(model: str, prompt: str, temperature: float, max_tokens: int, system_prompt: Optional[str]) -> str:
        """Call Anthropic API"""
        import anthropic
        
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        
        messages = [{"role": "user", "content": prompt}]
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt if system_prompt else "",
            messages=messages
        )
        
        return response.content[0].text
    
    @staticmethod
    def _call_deepseek(model: str, prompt: str, temperature: float, max_tokens: int, system_prompt: Optional[str]) -> str:
        """Call DeepSeek API (OpenAI-compatible)"""
        import openai
        
        client = openai.OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    @staticmethod
    def _call_google(model: str, prompt: str, temperature: float, max_tokens: int, system_prompt: Optional[str]) -> str:
        """Call Google Gemini API"""
        import google.generativeai as genai
        
        genai.configure(api_key=config.GOOGLE_API_KEY)
        
        model_instance = genai.GenerativeModel(model)
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = model_instance.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        
        return response.text

# Convenience function
def call_llm_with_fallback(prompt: str, models: list, temperature: float = 0.7, max_tokens: int = 4000, system_prompt: Optional[str] = None) -> str:
    """
    Try calling multiple models in order until one succeeds
    
    Args:
        prompt: The prompt to send
        models: List of model names to try
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        system_prompt: Optional system prompt
        
    Returns:
        The first successful response
        
    Raises:
        Exception if all models fail
    """
    last_error = None
    
    for model in models:
        try:
            return ModelFactory.call_llm(model, prompt, temperature, max_tokens, system_prompt)
        except Exception as e:
            last_error = e
            continue
    
    raise Exception(f"All models failed. Last error: {last_error}")
