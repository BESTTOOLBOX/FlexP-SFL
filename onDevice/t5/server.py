import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import DataCollatorWithPadding
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import pickle
# Load the GPT-2 model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval import evaluation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device setup complete:", device)
print("Available GPUs:", torch.cuda.device_count())

# Load the Flan-T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/")

split_start = [3, 3, 9, 6, 6]
split_end = [10, 10, 10, 10, 10]

with open("./client_0_top.pkl",'wb') as f:
    client_activations=pickle.load(f)
    
server_model=nn.Sequential(*list(model.decoder.block[split_start[0]:split_end[0]]))

def custom_forward(activations, server_model):
    hidden_state = activations
    for layer in server_model:
        hidden_state = layer(hidden_state, attention_mask=attention_mask)[0] 
    return hidden_state

overall_activations=[]
for batch in client_activations:
    activations = custom_forward(client_activations.to(device), server_model.to(device))  
    overall_activations.append(activations)
    break

print(overall_activations)

raise Exception("Good!")

