import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
import torch
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize model and tokenizer dictionaries
models = {}
tokenizers = {}

def get_available_models():
    """
    Get list of available models in the models directory.
    
    Returns:
        list: List of available model names
    """
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    return [d.name for d in models_dir.iterdir() if d.is_dir()]

def load_model(model_name):
    """
    Load a specific model and its tokenizer.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    try:
        # Check if model is already loaded
        if model_name in models and model_name in tokenizers:
            return models[model_name], tokenizers[model_name]
        
        model_path = Path("models") / model_name
        
        # Check if model directory exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        # Check if model files exist
        model_file = model_path / "model.safetensors"
        adapter_file = model_path / "adapter_model.safetensors"
        
        if not model_file.exists() and not adapter_file.exists():
            raise FileNotFoundError(f"No model file found at {model_path}. Expected either model.safetensors or adapter_model.safetensors")
        
        # Load tokenizer and model from local directory
        logger.info(f"Loading tokenizer for {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise Exception(f"Failed to load tokenizer for {model_name}: {str(e)}")
        
        logger.info(f"Loading model {model_name}...")
        
        # Load model configuration
        try:
            config = AutoConfig.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Failed to load model config: {str(e)}")
            raise Exception(f"Failed to load model config for {model_name}: {str(e)}")
        
        # Load model with appropriate configuration
        try:
            # Determine model type and load accordingly
            if "pythia" in model_name.lower():
                # Load base model first
                base_model = AutoModelForCausalLM.from_pretrained(
                    "EleutherAI/pythia-410m",
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                ).to("cpu")
                # Use base model directly instead of loading adapter
                model = base_model
            elif "gpt2" in model_name.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                ).to("cpu")
            elif "gptneo" in model_name.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Default loading for unknown model types
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise Exception(f"Failed to load model {model_name}: {str(e)}")
        
        # Store in dictionaries
        models[model_name] = model
        tokenizers[model_name] = tokenizer
        
        logger.info(f"Model {model_name} loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        return None, None

def unload_model(model_name):
    """
    Unload a model and its tokenizer to free up memory.
    
    Args:
        model_name (str): Name of the model to unload
    """
    if model_name in models:
        del models[model_name]
    if model_name in tokenizers:
        del tokenizers[model_name]
    torch.cuda.empty_cache()  # Clear GPU memory

def get_relevant_context(previous_content, max_tokens=800, model_name=None):
    """
    Get the most relevant context from the story, prioritizing recent content
    and important story elements.
    
    Args:
        previous_content (str): The full story content
        max_tokens (int): Maximum number of tokens to include in context
        model_name (str): Name of the model to use
        
    Returns:
        str: Relevant context for story continuation
    """
    if not previous_content:
        return ""
        
    try:
        # Load model if not already loaded
        model, tokenizer = load_model(model_name)
        if model is None or tokenizer is None:
            raise Exception(f"Failed to load model {model_name}")
            
        # Split content into paragraphs
        paragraphs = previous_content.split('\n\n')
        
        # Always include the last paragraph (most recent content)
        context = paragraphs[-1]
        current_tokens = len(tokenizer.encode(context))
        
        # Calculate how many paragraphs we can include
        remaining_tokens = max_tokens - current_tokens
        
        # If we have enough tokens, include more paragraphs
        if remaining_tokens > 0:
            # Start from the end and work backwards
            for para in reversed(paragraphs[:-1]):
                para_tokens = len(tokenizer.encode(para))
                if current_tokens + para_tokens > max_tokens:
                    break
                context = para + '\n\n' + context
                current_tokens += para_tokens
        
        return context
    except Exception as e:
        logger.error(f"Error in get_relevant_context: {str(e)}")
        # Return just the last paragraph if there's an error
        return paragraphs[-1] if paragraphs else ""

def generate_with_local_model(prompt, previous_content="", max_tokens=500, temperature=0.7, model_name=None):
    """
    Generate story content using the specified local model.
    
    Args:
        prompt (str): The prompt to continue the story from
        previous_content (str): The story content generated so far
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Controls randomness (0.0-1.0)
        model_name (str): Name of the model to use
        
    Returns:
        str: Generated story chunk
        
    Raises:
        Exception: If model loading or generation fails
    """
    try:
        # Load model if not already loaded
        model, tokenizer = load_model(model_name)
        if model is None or tokenizer is None:
            raise Exception(f"Failed to load model {model_name}")
        
        # Clean up the prompt by removing extra whitespace and newlines
        prompt = prompt.strip()
        
        # Get relevant context if there's previous content
        if previous_content:
            context = get_relevant_context(previous_content, model_name=model_name)
            # Use a more natural prompt format for continuation
            full_prompt = f"{context}\n\nContinue the story: {prompt}"
        else:
            # For new stories, use a more creative prompt
            full_prompt = f"Write a creative story: {prompt}"
            
        # Tokenize input
        try:
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        except Exception as e:
            raise Exception(f"Failed to tokenize input: {str(e)}")
        
        # Generate text with reduced batch size and memory usage
        with torch.no_grad():
            try:
                # Get generation config from model
                generation_config = model.generation_config
                
                # Update generation config with our parameters
                if "pythia" in model_name.lower():
                    # Generation config for Pythia with longer output
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=min(max_tokens, 200),  # Ensure minimum length
                        min_new_tokens=100,  # Force minimum new tokens
                        temperature=0.8,  # Slightly higher temperature for more creativity
                        do_sample=True,
                        top_p=0.92,  # Slightly higher top_p for more diversity
                        top_k=50,  # Add top_k sampling
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.3,  # Slightly reduced to allow more natural flow
                        no_repeat_ngram_size=2,  # Reduced to allow more natural repetition
                        min_length=100,  # Increased minimum length
                        max_length=inputs.input_ids.shape[1] + max_tokens,
                        num_return_sequences=1,
                        early_stopping=False,  # Disable early stopping to ensure full generation
                        length_penalty=1.0,  # Neutral length penalty
                        num_beams=1  # Use greedy decoding
                    )
                else:
                    # Original generation config for other models
                    generation_config.max_new_tokens = max_tokens
                    generation_config.min_new_tokens = 100  # Add minimum new tokens
                    generation_config.temperature = temperature
                    generation_config.do_sample = True
                    generation_config.top_p = 0.92
                    generation_config.top_k = 50
                    generation_config.repetition_penalty = 1.2
                    generation_config.num_beams = 1
                    generation_config.early_stopping = False
                    generation_config.no_repeat_ngram_size = 2
                    generation_config.length_penalty = 1.0
                    generation_config.min_length = 100
                    
                    # Generate with updated config
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                # Try with more conservative parameters
                try:
                    generation_config.max_new_tokens = min(max_tokens, 100)
                    generation_config.temperature = min(temperature, 0.7)
                    generation_config.repetition_penalty = 1.1
                    
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                except Exception as e2:
                    raise Exception(f"Failed to generate text even with conservative parameters: {str(e2)}")
        
        # Decode and return the generated text
        try:
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            if full_prompt in generated_text:
                new_content = generated_text.split(full_prompt)[-1].strip()
            else:
                # If prompt not found, try to find the last unique sentence
                sentences = generated_text.split('.')
                if len(sentences) > 1:
                    new_content = '.'.join(sentences[1:]).strip()
                else:
                    new_content = generated_text.strip()
            
            # For the first chunk, include the seed sentence with proper formatting
            if not previous_content:
                # Clean up the new content by removing any leading/trailing whitespace and extra newlines
                new_content = new_content.strip()
                # Ensure there's exactly one newline between the prompt and new content
                return f"{prompt}\n{new_content}"
            
            # For continuation, ensure we're not repeating the prompt
            if prompt in new_content:
                new_content = new_content.replace(prompt, "").strip()
            
            return new_content
            
        except Exception as e:
            raise Exception(f"Failed to extract new content: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in generate_with_local_model: {str(e)}")
        raise Exception(f"Error generating story: {str(e)}")

def generate_story_chunk(prompt, previous_content="", model_name=None):
    """
    Main function to generate story chunks, with fallback.
    
    Args:
        prompt (str): The prompt to continue the story from
        previous_content (str): The story content generated so far
        model_name (str): Name of the model to use
        
    Returns:
        str: Generated story chunk
    """
    try:
        if model_name is None:
            raise Exception("No model specified")
            
        model_path = Path("models") / model_name
        
        # Check if model directory exists and contains required files
        if not model_path.exists():
            raise Exception(f"Model directory not found: {model_path}")
            
        model_file = model_path / "model.safetensors"
        adapter_file = model_path / "adapter_model.safetensors"
        
        if not model_file.exists() and not adapter_file.exists():
            raise Exception(f"No model file found at {model_path}. Expected either model.safetensors or adapter_model.safetensors")
        
        try:
            # Try to generate with the model
            return generate_with_local_model(prompt, previous_content, model_name=model_name)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Model generation failed: {error_msg}")
            raise Exception(f"Failed to generate story: {error_msg}")
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in generate_story_chunk: {error_msg}")
        raise Exception(f"Failed to generate story: {error_msg}")

def generate_story_fallback(prompt, previous_content="", error_message=None):
    """
    Fallback function when model is not available or fails.
    
    Args:
        prompt (str): The prompt to continue the story from
        previous_content (str): The story content generated so far
        error_message (str): Optional error message to include in the fallback
        
    Returns:
        str: Generated story chunk
    """
    logger.warning(f"Using fallback story generation. Error: {error_message}")
    raise Exception(f"Model generation failed: {error_message}")