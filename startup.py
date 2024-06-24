from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, HfFolder
import os

if __name__ == "__main__":
    model_name = 'mistralai/Mixtral-8x22B-v0.1'
    
    # Set the Hugging Face token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        HfFolder.save_token(hf_token)
    else:
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set")

    # Download and save the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model.save_pretrained('/app/model')
    tokenizer.save_pretrained('/app/model')
