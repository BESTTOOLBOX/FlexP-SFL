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
import numpy as np
import time
from tqdm import tqdm

device = torch.device("cpu" if torch.cuda.is_available() else "cpu") #cuda:0
print("Device setup complete:", device)
print("Available GPUs:", torch.cuda.device_count())

client_num=1
client_model_start=2
client_model_end=10
IP_address='127.0.0.1'
IP_port=50002
Server_IP_address='127.0.0.1'
Server_IP_port=50010
print(f"Client No.: {client_num} Client_IP:{IP_address} Client_Port:{IP_port} Server_IP:{Server_IP_address} Server_Port:{Server_IP_port}")
tcp_tunnel_s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_tunnel_s.bind((IP_address, IP_port))
tcp_tunnel_s.listen(5)
symbol_act=0
symbol_mask=0
symbol_grad=0
criterion=torch.nn.CrossEntropyLoss()
lr=5e-5
batch_size_setting=1
server_stamp=0

def custom_forward(input_ids, attention_mask):
    global embedding_wte, embedding_wpe, client_model_top
    input_embeds = embedding_wte(input_ids) + embedding_wpe(torch.arange(input_ids.size(1)).unsqueeze(0))
    hidden_state = input_embeds
    for layer in client_model_top:
        hidden_state = layer(hidden_state, attention_mask=attention_mask)[0] 
    return hidden_state,attention_mask
    
def custom_forward_bottom(server_activation_batch, client_mask):
    global client_model_bottom, client_model_bottom_lnf, client_model_bottom_lmhead
    hidden_state = server_activation_batch
    for layer in client_model_bottom:
        hidden_state = layer(hidden_state, attention_mask=client_mask)[0] 
    hidden_state=client_model_bottom_lnf(hidden_state)
    hidden_state=client_model_bottom_lmhead(hidden_state)
    return hidden_state
    
def sendmsg(data, HOST, PORT):
    while 1:
        try:
            tcp_start_time=time.time()
            senddata = dill.dumps(data)
            size_of_x = sys.getsizeof(senddata)
            print(f"Size of Data: {size_of_x} bytes")
            tcp_tunnel_ss=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #print("1")
            tcp_tunnel_ss.connect((HOST, PORT))
            #print("2")
            tcp_tunnel_ss.sendall(senddata)
            #print("3")
            tcp_tunnel_ss.close()
            #print("4")
            tcp_end_time=time.time()
            print(f"TCP Success in {tcp_end_time-tcp_start_time} seconds")
            return 0
        except:
            print("Next Try in 2 seconds..."+str(HOST)+"//"+str(PORT))
            time.sleep(2)
    return 0

def comm_process_send(data, server_IP, server_port):
    print("Trasmitting...")
    size_of_x = sys.getsizeof(data)
    print(f"Size of Data: {size_of_x} bytes")
    sendmsg(data,server_IP,server_port)
    
def recvmsg(tcp_tunnel_s):
    global batch_size_setting
    while 1:
        try:
            tcp_start_time_recv=time.time()
            sock, addr = tcp_tunnel_s.accept()
            total_data = b''
            data = sock.recv(1024)
            total_data += data
            num = len(data)
            with tqdm(total=np.ceil(batch_size_setting*3000)) as bar:
                while len(data) > 0:
                    data = sock.recv(1024)
                    num += len(data)
                    total_data += data
                    bar.update(1)
            clientdata=total_data
            print("Receiving TCP from " + str(addr))
            clientdata=dill.loads(clientdata)
            tcp_end_time_recv=time.time()
            deal_recv(clientdata)
        except:
            pass
    return 0
    
def deal_recv(clientdata):
    global server_activation
    global server_mask
    global client_server_grad, server_stamp
    global symbol_act, symbol_mask, symbol_grad
    if clientdata['key']=='client_server_activation':
        server_activation=clientdata['data']
        server_stamp=clientdata['stamp']
        symbol_act=1
        return 0
    if clientdata['key']=='server_mask':
        server_mask=clientdata['data']
        symbol_mask=1
        return 0
    if clientdata['key']=='client_server_grad':
        client_server_grad=clientdata['data']
        symbol_grad=1
        return 0

def client_load_model(client_num, client_model_start, client_model_end):
    global client_model_top, tokenizer, embedding_wte, embedding_wpe, device, client_model_bottom, client_model_bottom_lnf, client_model_bottom_lmhead, lr, optimizer_top, optimizer_bottom
    with open("./model/client"+str(client_num)+"_top_"+str(client_model_start)+"_"+str(client_model_end)+".pkl",'rb') as f:
        client_model_top=pickle.load(f)
        client_model_top.to(device)
    with open("./tokenizer/tokenizer.pkl",'rb') as f:
        tokenizer=pickle.load(f)
    with open("./embedding/embedding_wte.pkl",'rb') as f:
        embedding_wte=pickle.load(f)
    with open("./embedding/embedding_wpe.pkl",'rb') as f:
        embedding_wpe=pickle.load(f)
    embedding_wte=embedding_wte.to(device)
    embedding_wpe=embedding_wpe.to(device)
    
    with open("./model/client"+str(client_num)+"_bottom_"+str(client_model_start)+"_"+str(client_model_end)+".pkl",'rb') as f:
        client_model_bottom=pickle.load(f)
        client_model_bottom.to(device)
    with open("./model/client"+str(client_num)+"_bottom_lnf_"+str(client_model_start)+"_"+str(client_model_end)+".pkl",'rb') as f:
        client_model_bottom_lnf=pickle.load(f)
        client_model_bottom_lnf.to(device)
    with open("./model/client"+str(client_num)+"_bottom_lmhead_"+str(client_model_start)+"_"+str(client_model_end)+".pkl",'rb') as f:
        client_model_bottom_lmhead=pickle.load(f)
        client_model_bottom_lmhead.to(device)

    optimizer_top = optim.Adam(client_model_top.parameters(), lr=lr)
    optimizer_bottom = optim.Adam(client_model_bottom.parameters(), lr=lr)
    print("Model Loaded!")

def client_load_data():
    global client_data, train_loader, batch_size_setting
    with open("./dataset/client"+str(client_num)+".pkl",'rb') as f:
        client_data=pickle.load(f)
    train_loader = DataLoader(client_data['train'], batch_size=batch_size_setting,  sampler=RandomSampler(client_data['train']))
    print("Data Loaded!")

def client_top_forward():
    global client_model_top, client_data, tokenizer, embedding_wte, embedding_wpe, device, train_loader, overall_activation_top, attention_mask, batch_size_setting, IP_address, IP_port, Server_IP_address, Server_IP_port
    #Client Top Activation
    overall_activation_top=[]
    overall_attention_mask=[]
    cnt=0
    for batch in train_loader:
        if cnt>=batch_size_setting:
            break
        temp_batch_ids=torch.stack(batch['input_ids'], axis=1)
        temp_batch_mask=torch.stack(batch['attention_mask'], axis=1)
        activation, attention_mask = custom_forward(temp_batch_ids.to(device), temp_batch_mask.to(device))
        overall_activation_top.append(activation)
        overall_attention_mask.append(attention_mask)
        cnt+=1
    #Client Top Comm
    package_overall_activation={'key':'client_top_activation','client':client_num,'host':IP_address,'data':overall_activation_top}
    comm_process_send(package_overall_activation, Server_IP_address, Server_IP_port)
    package_attention_mask={'key':'client_top_mask','client':client_num,'host':IP_address,'data':overall_attention_mask}
    comm_process_send(package_attention_mask, Server_IP_address, Server_IP_port)
    return 0

def client_bottom_forward():
    global server_activation, server_mask, overall_activation_bottom, device, client_model_bottom, client_model_bottom_lnf, client_model_bottom_lmhead
    #Client Bottom Activation
    overall_activation_bottom=[]
    for batch in range(len(server_activation)):
        server_activation[batch].requires_grad=True
        activation = custom_forward_bottom(server_activation[batch].to(device),server_mask[batch].to(device))  
        overall_activation_bottom.append(activation)
    return 0 

def client_bottom_back():
    global train_loader, overall_activation_bottom, criterion, server_activation, optimizer_bottom, batch_size_setting, IP_address, IP_port, Server_IP_address, Server_IP_port, server_stamp
    overall_grad_bottom=[]
    batch_i=0
    cnt=0
    for batch in train_loader:
        if cnt>=batch_size_setting:
            break
        print(overall_activation_bottom[batch_i].view(-1,overall_activation_bottom[batch_i].size(-1)).shape)
        print(torch.stack(batch['labels'], axis=1).view(-1).shape)
        loss=criterion(overall_activation_bottom[batch_i].view(-1,overall_activation_bottom[batch_i].size(-1)),torch.stack(batch['labels'], axis=1).view(-1).to(device))
        print(f"Loss: {loss.item()}")
        loss.backward(retain_graph=True) 
        grad=server_activation[batch_i].grad.detach()
        optimizer_bottom.step()
        overall_grad_bottom.append(grad)
        batch_i+=1
        cnt+=1
    #Client Bottom Comm Grad
    package_overall_grad={'key':'client_bottom_grad','client':client_num,'host':IP_address,'data':overall_grad_bottom,'stamp':server_stamp}
    comm_process_send(package_overall_grad,Server_IP_address, Server_IP_port)
    return 0

def client_top_back():
    global client_server_grad, optimizer_top, overall_activation_top, device
    for batch in range(len(client_server_grad)):
        overall_activation_top[batch].backward(client_server_grad[batch].to(device))
        optimizer_top.step()
    return 0

#TCP Receiver
thread_recvmsg_tcp=threading.Thread(target=recvmsg, args=(tcp_tunnel_s,))
thread_recvmsg_tcp.start()

if __name__ == '__main__':
    client_load_model(client_num, client_model_start, client_model_end)
    client_load_data()
    process_state=['Top_forward','Bottom_forward','Bottom_back','Top_back']
    round_cnt=0
    while True:
        round_cnt+=1
        symbol_act=0
        symbol_mask=0
        symbol_grad=0
        print(f"Round: {round_cnt}")
        for state in process_state:
            print(state)
            while True:
                if state=='Top_forward':
                    client_top_forward()
                    print("Top_forward Success")
                    break
                if state=='Bottom_forward' and symbol_act==1 and symbol_mask==1:
                    client_bottom_forward()
                    print("Bottom_forward Success")
                    break
                if state=='Bottom_back':
                    client_bottom_back()
                    print("Bottom_back Success")
                    break
                if state=='Top_back' and symbol_grad==1:
                    client_top_back()
                    print("Top_back Success")
                    break
                print('.', end="")
                time.sleep(1)
        


