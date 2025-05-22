import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model(model_name):
    """
    Test if a model can be loaded and used for generation.
    
    Args:
        model_name (str): Name of the model directory to test
        
    Returns:
        bool: True if model is valid, False otherwise
    """
    try:
        model_path = Path("models") / model_name
        
        # Check if model directory exists
        if not model_path.exists():
            logger.error(f"Model directory not found: {model_path}")
            return False
            
        # Check for adapter config to determine if it's a LoRA model
        is_lora = (model_path / "adapter_config.json").exists()
        
        if is_lora:
            logger.info(f"Detected LoRA adapter model: {model_name}")
            # Load adapter config
            adapter_config = PeftConfig.from_pretrained(model_path)
            base_model_name = adapter_config.base_model_name_or_path
            logger.info(f"Base model: {base_model_name}")
            
            # Load base model and tokenizer
            logger.info("Loading base model and tokenizer...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                device="cpu",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            # Load adapter
            logger.info("Loading adapter...")
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Regular model loading
            logger.info(f"Loading regular model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                device="cpu",
                trust_remote_code=True
            )
        
        # Test generation
        test_prompt = "Once upon a time"
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test generation successful for {model_name}")
        logger.info(f"Generated text: {generated_text}")
        
        # Clean up
        del model
        del tokenizer
        if is_lora:
            del base_model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing model {model_name}: {str(e)}")
        return False

def main():
    """Main function to test all models."""
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error("Models directory not found!")
        return
        
    model_dirs = [d.name for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        logger.error("No model directories found!")
        return
        
    logger.info(f"Found {len(model_dirs)} model directories: {model_dirs}")
    
    results = {}
    for model_name in model_dirs:
        logger.info(f"\nTesting model: {model_name}")
        is_valid = test_model(model_name)
        results[model_name] = is_valid
        
    # Print summary
    logger.info("\nTest Results Summary:")
    logger.info("-" * 50)
    for model_name, is_valid in results.items():
        status = "✅ Valid" if is_valid else "❌ Invalid"
        logger.info(f"{model_name}: {status}")

if __name__ == "__main__":
    main() 