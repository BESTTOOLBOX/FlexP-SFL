import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import DataCollatorWithPadding
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import pickle
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
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
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device setup complete:", device)
print("Available GPUs:", torch.cuda.device_count())

IP_address='127.0.0.1'
IP_port=50010
client_total_num=2
client_IP_address=['127.0.0.1','127.0.0.1']
client_IP_port=[50001,50002]
print(f"Server_IP:{IP_address} Server_Port:{IP_port} Total_client_number:{client_total_num}")
tcp_tunnel_s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_tunnel_s.bind((IP_address, IP_port))
tcp_tunnel_s.listen(5)
client_activation=[]
client_mask=[]
client_grad=[]
client_activation_num=[]
client_mask_num=[]
client_grad_num=[]
client_grad_stamp=[]
activation_wait_for_grad={}
lr=5e-5
split_start = [2,2]
split_end = [10,10]
batch_size_setting=1
temp_client_activation=[]
temp_client_mask=[]
temp_client_activation_num=[]
temp_client_mask_num=[]

def sendmsg(data, HOST, PORT):
    while 1:
        try:
            tcp_start_time=time.time()
            senddata = dill.dumps(data)
            tcp_tunnel_ss=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_tunnel_ss.connect((HOST, PORT))
            tcp_tunnel_ss.sendall(senddata)
            tcp_tunnel_ss.close()
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
            #print("1")
            total_data = b''
            data = sock.recv(1024)
            #print("2")
            total_data += data
            num = len(data)
            #print("3")
            with tqdm(total=np.ceil(batch_size_setting*3000)) as bar:
                while len(data) > 0:
                    data = sock.recv(1024)
                    num += len(data)
                    total_data += data
                    bar.update(1)
                    #print(data)
            clientdata=total_data
            print("Receiving TCP from " + str(addr))
            clientdata=dill.loads(clientdata)
            tcp_end_time_recv=time.time()
            deal_recv(clientdata)
        except:
            pass
    return 0
    
def deal_recv(clientdata):
    global temp_client_activation, client_grad, temp_client_mask, temp_client_activation_num, client_grad_num, temp_client_mask_num, client_grad_stamp
    if clientdata['key']=='client_top_activation':
        temp_client_activation.append(clientdata['data'])
        temp_client_activation_num.append(clientdata['client'])
        return 0
    if clientdata['key']=='client_top_mask':
        temp_client_mask.append(clientdata['data'])
        temp_client_mask_num.append(clientdata['client'])
        return 0
    if clientdata['key']=='client_bottom_grad':
        #print(clientdata)
        client_grad.append(clientdata['data'])
        client_grad_num.append(clientdata['client'])
        client_grad_stamp.append(clientdata['stamp'])
        #print(client_grad)
        #print(client_grad_num)
        #print(client_grad_stamp)
        return 0

def deal_recv_s():
    global temp_client_activation, client_activation, temp_client_activation_num, client_activation_num, temp_client_mask, client_mask, temp_client_mask_num, client_mask_num
    while True:
        if (len(temp_client_activation)>0) and (len(temp_client_mask)>0):
            

def server_load_model():
    global tokenizer, model, optimizer, lr, device
    tokenizer = AutoTokenizer.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    print("Model Loaded!")
    
def custom_forward(activations, server_model,client_masks):
    hidden_state = activations
    for layer in server_model:
        hidden_state = layer(hidden_state, attention_mask=client_masks)[0] 
    return hidden_state, client_masks

def server_forward():
    global client_activation, client_mask, client_activation_num, client_mask_num, split_start, split_end, device, batch_size_setting, IP_address, client_IP_address, client_IP_port, model, activation_wait_for_grad#, pop_client_activation
    while True:
        if len(client_mask)>0 and len(client_mask)==len(client_activation):
            if client_activation_num[0]!=client_mask_num[0]:
                continue
            print(f"Server Forward Start for Client:{client_mask_num[0]}")
            pop_client_activation=client_activation.pop(0)
            pop_client_mask=client_mask.pop(0)
            pop_client_num=client_mask_num.pop(0)
            pop_client_num=client_activation_num.pop(0)
            server_model=nn.Sequential(*list(model.transformer.h[split_start[pop_client_num]:split_end[pop_client_num]]))
            overall_activation=[]
            overall_attention_mask=[]
            cnt=0
            for batch in range(len(pop_client_activation)):
                if cnt>=batch_size_setting:
                    break
                pop_client_activation[batch].requires_grad=True
                activation, attention_mask = custom_forward(pop_client_activation[batch].to(device),server_model,pop_client_mask[batch].to(device))  
                overall_activation.append(activation)
                overall_attention_mask.append(attention_mask)
                cnt+=1
            stamp=str(time.time())+str(random.randint(0, 100))
            activation_wait_for_grad[stamp]=pop_client_activation
            package_overall_activation={'key':'client_server_activation','client':pop_client_num,'host':IP_address,'data':overall_activation,'stamp':stamp}
            comm_process_send(package_overall_activation, client_IP_address[pop_client_num], client_IP_port[pop_client_num])
            package_attention_mask={'key':'server_mask','client':pop_client_num,'host':IP_address,'data':overall_attention_mask}
            comm_process_send(package_attention_mask, client_IP_address[pop_client_num], client_IP_port[pop_client_num])
            print(f"Server Forward for Client:{pop_client_num} Success!")


def server_back():
    global client_grad, client_grad_num, model, client_grad_stamp, activation_wait_for_grad, optimizer, IP_address, client_IP_address, client_IP_port, device, batch_size_setting#, pop_client_activation
    while True:
        if len(client_grad)>0 and len(client_grad)==len(client_grad_num) and len(client_grad)==len(client_grad_stamp):
            print(f"Server Backward Start for Client:{client_grad_num[0]}")
            pop_client_num=client_grad_num.pop(0)
            pop_client_grad_stamp=client_grad_stamp.pop(0)
            pop_client_grad=client_grad.pop(0)
            pop_stamped_client_activation=activation_wait_for_grad[pop_client_grad_stamp]
            overall_grad=[]
            cnt=0
            for batch in range(len(pop_client_grad)):
                if cnt>=batch_size_setting:
                    break
                #print(pop_stamped_client_activation[batch])
                #print(pop_client_grad[batch])
                pop_stamped_client_activation[batch].backward(pop_client_grad[batch].to(pop_stamped_client_activation[batch].device)) #
                #print(pop_stamped_client_activation[batch])
                grad=pop_stamped_client_activation[batch].grad.detach() #_stamped
                optimizer.step()
                overall_grad.append(grad)
                cnt+=1
            package_overall_grad={'key':'client_server_grad','client':pop_client_num,'host':IP_address,'data':overall_grad}
            comm_process_send(package_overall_grad, client_IP_address[pop_client_num], client_IP_port[pop_client_num])
            print(f"Server Backward for Client:{pop_client_num} Success!")

#TCP Receiver
thread_recvmsg_tcp=threading.Thread(target=recvmsg, args=(tcp_tunnel_s,))
thread_recvmsg_tcp.start()

if __name__ == '__main__':
    server_load_model()
    print("Server Forward Watcher Start!")
    server_forward_thread=threading.Thread(target=server_forward, args=())
    server_forward_thread.start()
    print("Server Backward Watcher Start!")
    server_back_thread=threading.Thread(target=server_back, args=())
    server_back_thread.start()
