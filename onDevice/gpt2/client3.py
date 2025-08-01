import torch
from transformers import  Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, random_split
import random
import copy
import torch.nn as nn
import datasets
import pickle
import time
import socket
import threading
import json
import sys
import dill
import torch.optim as optim

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("Device setup complete:", device)
print("Available GPUs:", torch.cuda.device_count())

import numpy as np

client_num=0

with open("./model/client0_top_2_10.pkl",'rb') as f:
    client_model=pickle.load(f)
with open("./dataset/client0.pkl",'rb') as f:
    client_data=pickle.load(f)
with open("./tokenizer/tokenizer.pkl",'rb') as f:
    tokenizer=pickle.load(f)
with open("./embedding/embedding_wte.pkl",'rb') as f:
    embedding_wte=pickle.load(f)
with open("./embedding/embedding_wpe.pkl",'rb') as f:
    embedding_wpe=pickle.load(f)
    
embedding_wte=embedding_wte.to(device)
embedding_wpe=embedding_wpe.to(device)
def custom_forward(input_ids, client_model, attention_mask=None):
    input_embeds = embedding_wte(input_ids) + embedding_wpe(torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0))
    hidden_state = input_embeds
    for layer in client_model:
        hidden_state = layer(hidden_state, attention_mask=attention_mask)[0] 
    return hidden_state,attention_mask

print(client_data)

train_loader = DataLoader(client_data['train'], batch_size=1,  sampler=RandomSampler(client_data['train']))


overall_activations=[]
attention_masks=[]
for batch in train_loader:
    temp_batch_ids=torch.stack(batch['input_ids'], axis=1)
    temp_batch_mask=torch.stack(batch['attention_mask'], axis=1)
    activations, attention_mask = custom_forward(temp_batch_ids.to(device), client_model.to(device), temp_batch_mask.to(device))  
    overall_activations.append(activations)
    attention_masks.append(attention_mask)
    break
    
print(overall_activations)

#raise Exception("Good!")

#print("Activations shape:", activations.shape)
#print("Activations:", activations)
with open("./client_0_top.pkl",'wb') as f:
    pickle.dump(overall_activations,f)
with open("./client_0_top_mask.pkl",'wb') as f:
    pickle.dump(attention_mask,f)
    

"""
package_activations=dill.dumps(overall_activations)

def sendmsg(data, local_Host, local_Port, target_Host, target_Port):
    while 1:
        try:
            s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((target_Host, target_Port))
            s.sendall(data)
            print("Transmission Finished!")
            s.close()
            return 0
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(1)
    return 0
    
print("Trasmitting...")
size_of_x = sys.getsizeof(package_activations)
print(f"Size of package_activations: {size_of_x} bytes")
sendmsg(package_activations,'127.0.0.1',50001,'127.0.0.1',50000)

"""
def custom_forward_bottom(activations, client_model_bottom,client_model_bottom_lnf,client_model_bottom_lmhead, client_masks):
    hidden_state = activations
    for layer in client_model_bottom:
        hidden_state = layer(hidden_state, attention_mask=client_masks)[0] 
    hidden_state=client_model_bottom_lnf(hidden_state)
    hidden_state=client_model_bottom_lmhead(hidden_state)
    return hidden_state

with open("./client_0_server.pkl",'rb') as f:
    server_activation=pickle.load(f)
with open("./client_0_server_mask.pkl",'rb') as f:
    server_mask=pickle.load(f)
with open("./model/client0_bottom_2_10.pkl",'rb') as f:
    client_model_bottom=pickle.load(f)
with open("./model/client0_bottom_lnf_2_10.pkl",'rb') as f:
    client_model_bottom_lnf=pickle.load(f)
with open("./model/client0_bottom_lmhead_2_10.pkl",'rb') as f:
    client_model_bottom_lmhead=pickle.load(f)

criterion=torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(client_model_bottom.parameters(), lr=5e-5)


overall_activations_bottom=[]
for batch in range(len(server_activation)):
    server_activation[batch].requires_grad=True
    activations = custom_forward_bottom(server_activation[batch].to(device), client_model_bottom.to(device),client_model_bottom_lnf.to(device),client_model_bottom_lmhead.to(device),server_mask[batch].to(device))  
    overall_activations_bottom.append(activations)
    break

print(overall_activations_bottom)

overall_grad=[]
batch_i=0
for batch in train_loader:
    print(overall_activations_bottom[batch_i].view(-1,overall_activations_bottom[batch_i].size(-1)).shape)
    #print(overall_activations_bottom[batch_i])
    #print(batch['labels'])
    #print(batch['input_ids'])
    print(torch.stack(batch['labels'], axis=1).view(-1).shape)
    #print(torch.stack(batch['input_ids'], axis=1))
    #print(torch.stack(batch['attention_mask'], axis=1))
    loss=criterion(overall_activations_bottom[batch_i].view(-1,overall_activations_bottom[batch_i].size(-1)),torch.stack(batch['labels'], axis=1).view(-1).to(device))
    print(f"Loss: {loss.item()}")
    loss.backward(retain_graph=True) 
    grad=server_activation[batch_i].grad.detach()
    print(grad)
    optimizer.step()
    overall_grad.append(grad)
    batch_i+=1
    break
    
with open("./client_0_bottom_grad.pkl",'wb') as f:
    pickle.dump(overall_grad,f)
    
with open("./client_0_server_grad.pkl",'rb') as f:
    client_server_grad=pickle.load(f)

for batch in range(len(client_server_grad)):
    overall_activations[batch].backward(client_server_grad[batch])
    optimizer.step()
    break

print("Success")