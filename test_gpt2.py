from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
tokenizer = GPT2Tokenizer.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/")
model = GPT2LMHeadModel.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/").to(device)

input_text = "Once upon a time in a distant galaxy"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

output = model.generate(
    input_ids,
    max_length=100, 
    num_return_sequences=1,  
    no_repeat_ngram_size=2,  
    do_sample=True,  
    top_k=50,  
    top_p=0.95,  
    temperature=0.7  
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)