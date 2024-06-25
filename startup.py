from huggingface_hub import HfApi, HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import time
import os
def generate_text(prompt, model_path='/app/model', max_length=50):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text
    start_time = time.time()  # Start timing
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_p=0.95,
            top_k=50
        )
    end_time = time.time()  # End timing

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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

    # Download and save the model and tokenizer

    # Define the directory where the model and tokenizer are saved
    model_directory = "/app/model"
    # Debugging output
    print(f"Contents of {model_directory}:")
    print(os.listdir(model_directory))
    
    # Start timing the loading process
    print("STARTING MODEL LOAD FROM DISK")
    load_time = time.time()

    # Load the configuration
    config = AutoConfig.from_pretrained(model_directory)

    # Load the model and tokenizer from the specified directory
    model = AutoModelForCausalLM.from_pretrained(model_directory, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)

    # Calculate the time taken to load the model and tokenizer
    load_end = time.time() - load_time
    print(f"Loading time took: {load_end}")


    prompt_1 = "Hello, what is your name?"
    text_1, time_1 = generate_text(prompt_1)
    print(f"Prompt 1 took {time_1} sec")
    print(f"Response 1: {text_1}")

    prompt_2 = "Create a Python function that takes an integer as an argument and returns 'Even' for even numbers or 'Odd' for odd numbers."
    text_2, time_2 = generate_text(prompt_2)
    print(f"Prompt 2 took {time_2} sec")
    print(f"Response 2: {text_2}")

    prompt_3 = pandas_problem = """input df = [
    [1, 2, 3],
    [4, 5, 6]
    ]
    column_input = ['A', 'B', 'C']
    Create a Python fuction function that returns a new pandas. DataFrame object with same data as input_df, but now its column names changed to the sequence in "column_input". You must not modify the original input."""
    text_3, time_3 = generate_text(prompt_3)
    print(f"Prompt 3 took {time_3} sec")
    print(f"Response 3: {text_3}")

    prompt_4 = """Create an SQLite function that returns a dataset with two columns: number and is_even, where number contains the original input value, and is_even containing "Even" or "Odd" depending on number column values."""
    text_4, time_4 = generate_text(prompt_4)
    print(f"Prompt 4 took {time_4} sec")
    print(f"Response 4: {text_4}")