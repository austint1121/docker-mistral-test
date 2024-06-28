from huggingface_hub import HfApi, HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import time
import os
from accelerate import infer_auto_device_map, dispatch_model, init_empty_weights
import safetensors.torch
from accelerate.utils import BnbQuantizationConfig



def generate_text(model, tokenizer, prompt, model_path='/app/model', max_length=50):
    # Load the model and tokenizer
    # Generate text
    start_time = time.time()  # Start timing
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=500)
    

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_time = time.time()  # End timing
    generation_time = end_time - start_time  # Calculate the time taken

    return generated_text, generation_time

if __name__ == "__main__":
    model_name = 'mistralai/Mixtral-8x22B-v0.1'
    
    # Set the Hugging Face token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        HfFolder.save_token(hf_token)
    else:
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set")

    # Define the directory where the model and tokenizer are saved
    model_directory = "/app/model"
    offload_path = "/app/offload"
    # bnb quantizatin
   
    # Start timing the loading process
    print("STARTING MODEL LOAD FROM DISK")
    load_time = time.time()

    # Load the configuration
    # config = AutoConfig.from_pretrained(model_directory)
    # bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, llm_int8_threshold = 6)
    model = AutoModelForCausalLM.from_pretrained(model_directory, low_cpu_mem_usage=True, device_map='auto', offload_folder = offload_path, torch_dtype=torch.float16)
    
    # model = AutoModelForCausalLM.from_config(config)
    # device_map = infer_auto_device_map(model, max_memory={"cpu": "300GiB", "cuda": "23GiB"})
    # model = dispatch_model(model, device_map=device_map)
    # model = model.from_pretrained(model_directory, load_in_4bit=True)
    # Load the model and tokenizer from the specified directory
    tokenizer = AutoTokenizer.from_pretrained(model_directory)


# Dispatch the model according to the device map

    # Calculate the time taken to load the model and tokenizer
    load_end = time.time() - load_time
    print(f"Loading time took: {load_end}")


    prompt_1 = "Hello, what is your name?"
    text_1, time_1 = generate_text(model, tokenizer, prompt_1)
    print(f"Prompt 1 took {time_1} sec")
    print(f"Response 1: {text_1}")

    prompt_2 = "Create a Python function that takes an integer as an argument and returns 'Even' for even numbers or 'Odd' for odd numbers."
    text_2, time_2 = generate_text(model, tokenizer,prompt_2)
    print(f"Prompt 2 took {time_2} sec")
    print(f"Response 2: {text_2}")

    prompt_3 = pandas_problem = """input df = [
    [1, 2, 3],
    [4, 5, 6]
    ]
    column_input = ['A', 'B', 'C']
    Create a Python fuction function that returns a new pandas. DataFrame object with same data as input_df, but now its column names changed to the sequence in "column_input". You must not modify the original input."""
    text_3, time_3 = generate_text(model, tokenizer,prompt_3)
    print(f"Prompt 3 took {time_3} sec")
    print(f"Response 3: {text_3}")

    prompt_4 = """Create an SQLite function that returns a dataset with two columns: number and is_even, where number contains the original input value, and is_even containing "Even" or "Odd" depending on number column values."""
    text_4, time_4 = generate_text(model, tokenizer,prompt_4)
    print(f"Prompt 4 took {time_4} sec")
    print(f"Response 4: {text_4}")