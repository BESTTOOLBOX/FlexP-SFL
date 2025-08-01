import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import DataCollatorWithPadding
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import pickle
import torch.optim as optim
# Load the GPT-2 model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval import evaluation

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("Device setup complete:", device)
print("Available GPUs:", torch.cuda.device_count())

# Load the Flan-T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/")

split_start = [2, 2, 2, 3, 3]
split_end = [10, 10, 10, 10, 10]

with open("./client_0_top.pkl",'rb') as f:
    client_activations=pickle.load(f)
with open("./client_0_top_mask.pkl",'rb') as f:
    client_masks=pickle.load(f)
    
server_model=nn.Sequential(*list(model.transformer.h[split_start[0]:split_end[0]]))
optimizer = optim.Adam(server_model.parameters(), lr=5e-5)

def custom_forward(activations, server_model,client_masks):
    hidden_state = activations
    for layer in server_model:
        hidden_state = layer(hidden_state, attention_mask=client_masks)[0] 
    return hidden_state, client_masks

overall_activations=[]
attention_masks=[]
for batch in range(len(client_activations)):
    client_activations[batch].requires_grad=True
    activations, attention_mask = custom_forward(client_activations[batch].to(device), server_model.to(device),client_masks[batch].to(device))  
    overall_activations.append(activations)
    attention_masks.append(attention_mask)
    break

print(overall_activations)

with open("./client_0_server.pkl",'wb') as f:
    pickle.dump(overall_activations,f)
with open("./client_0_server_mask.pkl",'wb') as f:
    pickle.dump(attention_mask,f)
    
with open("./client_0_bottom_grad.pkl",'rb') as f:
    client_bottom_grad=pickle.load(f)
    
overall_grad=[]
for batch in range(len(client_bottom_grad)):
    client_activations[batch].backward(client_bottom_grad[batch])
    grad=client_activations[batch].grad.detach()
    optimizer.step()
    overall_grad.append(grad)
    break
print(overall_grad)

with open("./client_0_server_grad.pkl",'wb') as f:
    pickle.dump(overall_grad,f)