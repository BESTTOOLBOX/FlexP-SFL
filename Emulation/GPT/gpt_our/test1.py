import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import json

from eval import evaluation  # 确保该模块已正确实现并可访问

# Constants
NUM_CLIENTS = 10
NUM_ROUNDS = 2000
BATCH_SIZE = 4
LAMBDA_KL = 0.25  # KL 散度的权重
GRAD_CLIP = 1.0    # 梯度裁剪阈值

# Define split points for 10 clients, using client_id // 2 to repeat first 5 splits
original_split_start = [3, 3, 9, 6, 6]  # Example split points for first 5 clients
original_split_end = [22, 22, 22, 22, 22]  # Assuming GPT-2 has 24 transformer blocks (h[0] to h[23])

split_start = [original_split_start[i // 2] for i in range(NUM_CLIENTS)]  # [3,3,9,6,6,3,3,9,6,6]
split_end = [original_split_end[i // 2] for i in range(NUM_CLIENTS)]      # [22,22,22,22,22,22,22,22,22,22]



# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device setup complete:", device)
print("Available GPUs:", torch.cuda.device_count())

# Load the GPT-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")

# Assign pad token
tokenizer.pad_token = tokenizer.eos_token

# Function to print memory usage
def print_memory_usage():
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # Convert to MB
        reserved_memory = torch.cuda.memory_reserved(device) / 1024 ** 2  # Convert to MB
        print(f"Allocated memory: {allocated_memory:.2f} MB")
        print(f"Reserved memory: {reserved_memory:.2f} MB")
    else:
        print("CUDA is not available.")

# Function to print model size
def print_model_size(model):
    param_size = sum(p.numel() for p in model.parameters()) * 4 / 1024**2  # Assuming float32
    print(f"Model size: {param_size:.2f} MB")

# Load client datasets
with open("../client_datasets.json", "r") as file:
    client_datasets = json.load(file)
# 4.30.2 1.13
# Duplicate client datasets to have 10 clients, with clients 6-10 copying clients 1-5
original_num_clients = len(client_datasets)
duplicated_client_datasets = copy.deepcopy(client_datasets)
for i in range(original_num_clients, NUM_CLIENTS):
    duplicated_client_datasets.append(copy.deepcopy(client_datasets[i - original_num_clients]))

client_datasets = duplicated_client_datasets

# Preprocessing function for GPT-2
def preprocess_mmlu_gpt(examples):
    inputs = []
    for question, choices, answer in zip(examples['question'], examples['choices'], examples['answer']):
        # Construct input prompt with correct answer
        answer_letter = f"{chr(65 + int(answer))}"  # Convert numerical answer to letter (0 -> A, 1 -> B, etc.)
        prompt = (f"Question: {question}. Choices: A) {choices[0]}, B) {choices[1]}, "
                  f"C) {choices[2]}, D) {choices[3]}. Output a single letter. Correct Answer: {answer_letter}")
        
        # Tokenize prompt with fixed max_length=1024
        tokenized_prompt = tokenizer(prompt, max_length=1024, truncation=True, padding="max_length")
        
        # Construct labels, set prompt part to -100 to ignore in loss computation
        labels = tokenized_prompt['input_ids'].copy()
        labels[:-1] = [-100] * (len(labels) - 1)  # Only the last token is used for loss
        
        # Ensure the length is 1024
        assert len(labels) == 1024, f"Labels length: {len(labels)} for prompt: {prompt}"
        
        # Combine prompt and labels
        tokenized_input = {
            'input_ids': tokenized_prompt['input_ids'],
            'attention_mask': tokenized_prompt['attention_mask'],
            'labels': labels
        }
        
        inputs.append(tokenized_input)
    
    return {
        'input_ids': [item['input_ids'] for item in inputs],
        'attention_mask': [item['attention_mask'] for item in inputs],
        'labels': [item['labels'] for item in inputs]
    }

# Preprocess datasets for all clients
tokenized_datasets = []
for i, dataset in enumerate(client_datasets):
    train_dataset = Dataset.from_dict(dataset["train"])
    test_dataset = Dataset.from_dict(dataset["test"])
    
    # Apply preprocessing
    tokenized_train = train_dataset.map(preprocess_mmlu_gpt, batched=True)
    tokenized_test = test_dataset.map(preprocess_mmlu_gpt, batched=True)
    
    # 查看当前的列名
    print(f"Client {i + 1} - Train columns before removing:", tokenized_train.column_names)
    print(f"Client {i + 1} - Test columns before removing:", tokenized_test.column_names)

    # Remove unwanted columns (assuming 'question', 'choices', 'answer' are present)
    tokenized_train = tokenized_train.remove_columns([col for col in tokenized_train.column_names 
                                                    if col not in ["input_ids", "attention_mask", "labels"]])
    tokenized_test = tokenized_test.remove_columns([col for col in tokenized_test.column_names
                                                    if col not in ["input_ids", "attention_mask", "labels"]])

    print(f"Client {i + 1} - Train columns after removing:", tokenized_train.column_names)
    print(f"Client {i + 1} - Test columns after removing:", tokenized_test.column_names)

    # 设置为torch格式
    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    tokenized_datasets.append({"train": tokenized_train, "test": tokenized_test})

# Define TransformerBlockWrapper class
class TransformerBlockWrapper(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, hidden_states, attention_mask=None):
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
        return hidden_states


# Define FullModel class
class FullModel(nn.Module):
    def __init__(self, wte, wpe, client_base, client_top, server_shared, client_bottom, split_start):
        super().__init__()
        self.wte = wte
        self.wpe = wpe
        self.client_base = client_base
        self.client_top = client_top
        self.server_shared = server_shared
        self.client_bottom = client_bottom
        self.split_start = split_start

    def forward(self, input_ids, attention_mask=None):
    # 将 input_ids 转换为 embeddings
        inputs_embeds = self.wte(input_ids) + self.wpe(torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0))

        # 1. 计算Z_k: 输入 -> client_base(0:3) -> client_top(3:split_start)
        Z_k_base = self.client_base(inputs_embeds, attention_mask=attention_mask)
        Z_k = self.client_top(Z_k_base, attention_mask=attention_mask) 
        # 此时Z_k到达了split_start层的输出

        # 2. 计算Z_k_prime: 输入 -> client_base(0:3) -> server_shared(3:split_start)
        Z_k_prime = self.client_base(inputs_embeds, attention_mask=attention_mask)
        needed_layers_for_prime = self.split_start - global_min_start  # split_start - 3
        for idx in range(needed_layers_for_prime):
            Z_k_prime = self.server_shared.blocks[idx](Z_k_prime, attention_mask=attention_mask)[0]
        # 此时Z_k_prime到达split_start层的表示，但走的是client_base + server_shared路线

        # 3. 最终outputs: 
        # 根据您的要求，最终输出需要从Z_k出发，继续经过 server_shared[(split_start - global_min_start):] 到 global_end，
        # 再通过client_bottom。
        # Z_k已经到达split_start层的表示，需要从server_shared剩余部分继续处理
        needed_layers_for_output = (global_end - global_min_start) - needed_layers_for_prime
        # 也就是 server_shared中从[needed_layers_for_prime : end]的层要继续处理Z_k
        Z_output = Z_k
        start_idx_for_output = needed_layers_for_prime
        for idx in range(start_idx_for_output, start_idx_for_output + needed_layers_for_output):
            Z_output = self.server_shared.blocks[idx](Z_output, attention_mask=attention_mask)[0]

        # 经过server_shared剩余部分后，通过client_bottom
        outputs = self.client_bottom(Z_output, attention_mask=attention_mask)

        return outputs, Z_k, Z_k_prime


class ClientBottom(nn.Module):
    def __init__(self, blocks, ln_f, lm_head):
        super(ClientBottom, self).__init__()
        self.blocks = TransformerBlockWrapper(blocks)  # 自定义的 TransformerBlockWrapper
        self.ln_f = ln_f  # 层归一化
        self.lm_head = lm_head  # 语言模型头

    def forward(self, hidden_states, attention_mask=None):
        # 通过剩余的 Transformer 层
        hidden_states = self.blocks(hidden_states, attention_mask=attention_mask)
        # 通过层归一化
        hidden_states = self.ln_f(hidden_states)
        # 通过语言模型头，生成 logits
        logits = self.lm_head(hidden_states)
        return logits
    
global_min_start = 2
global_end = 22

# 定义 server_shared = h[3:22]
server_shared_blocks = list(model.transformer.h[global_min_start:global_end])
server_shared = nn.Module()  # 用于将 blocks 封装在 Module 里
server_shared.blocks = nn.ModuleList(server_shared_blocks)
server_shared.to(device)


# Define split_gpt2_model function
def split_gpt2_model(model, split_s, split_e):
    # client_base: h[0:global_min_start]
    client_base_blocks = list(model.transformer.h[:global_min_start])
    client_base = TransformerBlockWrapper(client_base_blocks)

    # client_top: h[0:split_s]
    client_top_blocks = list(model.transformer.h[global_min_start:split_s])
    client_top = TransformerBlockWrapper(client_top_blocks)

    # client_bottom: h[split_e:]
    client_bottom_blocks = list(model.transformer.h[split_e:])
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head
    client_bottom = ClientBottom(client_bottom_blocks, ln_f, lm_head)

    return client_base, client_top, client_bottom

# Initialize client models
client_models_top = []
client_models_base = []
# server_shared_models = []
client_models_bottom = []
full_models = []

for i in range(NUM_CLIENTS):
    client_base, client_top, client_bottom = split_gpt2_model(model, split_start[i], split_end[i])
    
    client_base.to(device)
    client_top.to(device)
    client_bottom.to(device)
    
    client_models_base.append(client_base)
    client_models_top.append(client_top)
    client_models_bottom.append(client_bottom)
    
    # Initialize FullModel for each client
    full_model = FullModel(model.transformer.wte, model.transformer.wpe, client_base, client_top, server_shared, client_bottom, split_start[i]).to(device)
    full_models.append(full_model)

# Initialize optimizers per client
lr=5e-8
server_optimizers = torch.optim.Adam(server_shared.parameters(), lr=lr)
client_bottom_optimizers = [torch.optim.Adam(client_models_bottom[i].parameters(), lr=lr) for i in range(NUM_CLIENTS)]
client_top_optimizers = [torch.optim.Adam(client_models_top[i].parameters(), lr=lr) for i in range(NUM_CLIENTS)]
client_base_optimizers = [torch.optim.Adam(client_models_base[i].parameters(), lr=lr) for i in range(NUM_CLIENTS)]

# Define loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer, padding=True)

# Define DataLoaders for clients
client_dataloaders = []
for i in range(NUM_CLIENTS):
    client_train_dataset = tokenized_datasets[i]['train']
    client_dataloader = DataLoader(
        client_train_dataset, 
        sampler=RandomSampler(client_train_dataset),
        batch_size=1
        #,
        #collate_fn=data_collator  # 至关重要
    )
    client_dataloaders.append(client_dataloader)

# Initial evaluation
all_losses = []
all_accuracies = []
initial_losses = []
initial_accuracies = []

import os
import csv

csv_filename_acc = "all_accuracies.csv"
csv_filename_lss = "all_losses.csv"

if os.path.exists(csv_filename_acc):
    os.remove(csv_filename_acc)
if os.path.exists(csv_filename_lss):
    os.remove(csv_filename_lss)

print("Initial evaluation of clients:")
for client_num in range(NUM_CLIENTS):
    # full_model already exists as full_models[client_num]
    full_model = full_models[client_num]
    full_model.eval()
    
    # Compute initial evaluation
    with torch.no_grad():
        acc = evaluation(tokenized_datasets[client_num]['test'], full_model, tokenizer)
    
    # Store initial accuracies
    initial_accuracies.append(acc)
    print(f"Client {client_num + 1} - Initial Accuracy: {acc}")

all_accuracies.append(initial_accuracies)

with open(csv_filename_acc, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([accuracy] for accuracy in initial_accuracies)  # 每个元素占一行


# Training loop
for round_num in range(NUM_ROUNDS):
    print(f"\n=== Communication Round {round_num + 1} ===")
    round_losses = []
    round_accuracies = []
    
    for client_id in range(NUM_CLIENTS):
        print(f"--- Training Client {client_id + 1} ---")
        client_dataloader = client_dataloaders[client_id]
        sample_count = 0 
        # Set model to train mode
        full_models[client_id].train()
        for batch in client_dataloader:
            sample_count += 1
            if (sample_count >= BATCH_SIZE): 
                break
            input_ids = batch['input_ids'].to(device)
            #print("??")
            # print(input_ids)

            attention_mask = batch['attention_mask'].to(device).bool()
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            #print(attention_mask)
            #print(f"attention_mask dtype after conversion: {attention_mask.dtype}")  # 添加打印以确认类型
            
            labels = batch['labels'].to(device)
            
            # Zero gradients
            server_optimizers.zero_grad()
            client_bottom_optimizers[client_id].zero_grad()
            client_top_optimizers[client_id].zero_grad()
            client_base_optimizers[client_id].zero_grad()
            
            # Forward pass
            outputs, Z_k, Z_prime_k = full_models[client_id](input_ids, attention_mask=attention_mask)
            logits = outputs  # 直接将 outputs 当作 logits 使用
            loss_ce = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            
            # Compute KL Divergence between client AL outputs and server SL outputs
            P_client = torch.softmax(Z_k, dim=-1)
            P_server = torch.softmax(Z_prime_k.detach(), dim=-1)
            
            kl_div = nn.KLDivLoss(reduction='batchmean')(torch.log(P_client + 1e-10), P_server)
            
            # Total loss
            total_loss = loss_ce + LAMBDA_KL * kl_div
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping (optional, good practice)
            torch.nn.utils.clip_grad_norm_(full_models[client_id].parameters(), GRAD_CLIP)
            
            # Update parameters
            client_bottom_optimizers[client_id].step()
            server_optimizers.step()
            client_top_optimizers[client_id].step()
            client_base_optimizers[client_id].step()
            
            # Collect loss
            round_losses.append(total_loss.item())
        
        if (round_num % 10 == 9):
            with torch.no_grad():
                full_models[client_id].eval()
                acc = evaluation(tokenized_datasets[client_id]['test'], full_models[client_id], tokenizer)
            round_accuracies.append(acc)
            print(f"Client {client_id + 1} - Round {round_num + 1} Accuracy: {acc}")
    
    all_losses.append(round_losses)
        
    if (round_num % 10 == 9):
        all_accuracies.append(round_accuracies)

    if (round_num % 10 == 9): 
        with open(csv_filename_acc, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([accuracy] for accuracy in round_accuracies)  # 每个元素占一行
    with open(csv_filename_lss, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([accuracy] for accuracy in round_losses)  # 每个元素占一行

print("\nTraining completed.")
print("All losses:", all_losses)
print("All accuracies:", all_accuracies)

print("Federated Split Learning with SFT, Weight Updating, and Personalized Evaluation completed.")
