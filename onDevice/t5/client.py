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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device setup complete:", device)
print("Available GPUs:", torch.cuda.device_count())

import numpy as np

client_num=0

with open("./model/client0_top_3_10.pkl",'rb') as f:
    client_model=pickle.load(f)
with open("./dataset/client0.pkl",'rb') as f:
    client_data=pickle.load(f)
with open("./tokenizer/tokenizer.pkl",'rb') as f:
    tokenizer=pickle.load(f)
with open("./embedding/embedding.pkl",'rb') as f:
    embedding=pickle.load(f)
    
embedding_layer = nn.Embedding.from_pretrained(embedding, freeze=False)

#print(client_data['train'].shape)

class SubT5Encoder(nn.Module):
    def __init__(self, encoder_layers, embedding_layer):
        super(SubT5Encoder, self).__init__()
        self.embedding_layer = embedding_layer
        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, input_ids, attention_mask=None):
        #print("0")
        hidden_states = self.embedding_layer(input_ids)
        #attention_mask = self.embedding_layer(attention_mask)
        #print("1")
        for layer in self.encoder_layers:
            #print("2")
            #print(hidden_states.shape)
            #print(attention_mask.shape)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
            #print("3")
        return hidden_states

sub_encoder = SubT5Encoder(
    encoder_layers=client_model,
    embedding_layer=embedding_layer
).to(device)

#print(sub_encoder)

#def collate_fn(batch):
#    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long).to(device)
#    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long).to(device)
#    return input_ids, attention_mask
train_loader = DataLoader(client_data['train'], batch_size=1,  sampler=RandomSampler(client_data['train'])) #, collate_fn=collate_fn
#print(train_loader)
#print("!!!!!!!!!!!")
#for i in train_loader: #
#    print(i)
#print("!!!!!!!!!!!")

#input_ids = torch.tensor(client_data['train']['input_ids'], dtype=torch.long).to(device)
#attention_mask = torch.tensor(client_data['train']['attention_mask'], dtype=torch.long).to(device)
overall_activations=[]
with torch.no_grad():
    for batch in train_loader:
        temp_batch_ids=torch.stack(batch['input_ids'], axis=1)
        temp_batch_mask=torch.stack(batch['attention_mask'], axis=1)
        #print(temp_batch_mask)
        #print(temp_batch_ids.shape)
        #print(temp_batch_mask.shape)
        activations = sub_encoder(temp_batch_ids.to(device), temp_batch_mask.to(device)) #/home/houyz/anaconda3/envs/SplitFM/lib/python3.7/site-packages/transformers/models/t5/modeling_t5.py", line 550
        #print("Activations shape:", activations.shape)
        #print("Activations:", activations)
#        break 
        overall_activations.append(activations)
        break

#print("Activations shape:", activations.shape)
#print("Activations:", activations)
with open("./client_0_top.pkl",'wb') as f:
    pickle.dump(overall_activations,f)

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