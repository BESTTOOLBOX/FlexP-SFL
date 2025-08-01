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

# We use left padding for the gpt-2 model
tokenizer = AutoTokenizer.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device setup complete:", device)
print("Available GPUs:", torch.cuda.device_count())


def print_memory_usage():
    allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # Convert to MB
    reserved_memory = torch.cuda.memory_reserved(device) / 1024 ** 2  # Convert to MB
    print(f"Allocated memory: {allocated_memory:.2f} MB")
    print(f"Reserved memory: {reserved_memory:.2f} MB")


def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(f"Model size: {size_all_mb:.2f} MB")


tokenizer.pad_token = tokenizer.eos_token

#mmlu_tasks = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge']
import json

# 从 JSON 文件加载数据
with open("./client_datasets.json", "r") as file:
    client_datasets = json.load(file)

# 检查导入的数据结构
for i, dataset in enumerate(client_datasets):
    print(f"Task {i+1} - Train size: {len(dataset['train']['question'])}, Test size: {len(dataset['test']['question'])}")

#raise Exception("Datasets Good!")

# GPT 的预处理逻辑
# GPT 的预处理逻辑

def preprocess_mmlu_gpt(examples):
    inputs = []

    for question, choices, answer in zip(examples['question'], examples['choices'], examples['answer']):
        # 构造输入提示，添加正确答案
        answer_letter = f"{chr(65 + int(answer))}"
        prompt = f"Question: {question}. Choices: A) {choices[0]}, B) {choices[1]}, C) {choices[2]}, D) {choices[3]}. Output a single letter. Correct Answer: {answer_letter}"

        # 对 prompt 进行编码
        tokenized_prompt = tokenizer(prompt, max_length=1024, truncation=True, padding="max_length")

        # 构造 labels，将 prompt 部分设置为 -100，避免计算它的损失
        labels = tokenized_prompt['input_ids'].copy()
        # set labels all to -100 except the last token
        labels[:-1] = [-100] * (len(labels) - 1)

        # print(tokenizer.decode(labels[-1]))

        # 将 prompt 和 labels 合并为最终的输入
        tokenized_input = {
            'input_ids': tokenized_prompt['input_ids'],
            'attention_mask': tokenized_prompt['attention_mask'],
            'labels': labels
        }

        # 收集每个样本的输入
        inputs.append(tokenized_input)

    # 将输入列表转化为单一的字典格式，以便 .map() 函数使用
    return {
        'input_ids': [item['input_ids'] for item in inputs],
        'attention_mask': [item['attention_mask'] for item in inputs],
        'labels': [item['labels'] for item in inputs]
    }

# 对每个数据集的 "train" 和 "test" 部分应用 .map()，适用于 GPT
from datasets import Dataset
tokenized_datasets = []
for dataset in client_datasets:
    train_dataset = Dataset.from_dict(dataset["train"])
    test_dataset = Dataset.from_dict(dataset["test"])

    # 使用 preprocess_mmlu 函数预处理数据
    tokenized_train = train_dataset.map(preprocess_mmlu_gpt, batched=True)
    tokenized_test = test_dataset.map(preprocess_mmlu_gpt, batched=True)
    tokenized_datasets.append({"train": tokenized_train, "test": tokenized_test})

# print(tokenized_datasets[0]['test'][0])
client_data_save_path='./dataset/'
client_data_save_name='client'
for i in range(len(tokenized_datasets)-1):
    with open(client_data_save_path+client_data_save_name+str(i)+'.pkl','wb') as f:
        pickle.dump(tokenized_datasets[i],f)

num_clients = 5


# Split the model into three parts: client_top, server_intermediate, client_bottom
def split_gpt2_model(model, split_start, split_end):
    client_top = nn.Sequential(*list(model.transformer.h[:split_start]))
    client_bottom = nn.Sequential(*list(model.transformer.h[split_end:]))
    client_bottom_lnf=nn.Sequential(model.transformer.ln_f)
    client_bottom_lmhead=nn.Sequential(model.lm_head)
    return client_top, client_bottom, client_bottom_lnf, client_bottom_lmhead


# Example split: first 2 layers are client top, last 2 layers are client bottom, others are server
split_start = [2, 2, 2, 3, 3]
split_end = [10, 10, 10, 10, 10]

# Initialize client models and server's intermediate part
client_models_top = [copy.deepcopy(model) for _ in range(5)]
client_models_bottom = [copy.deepcopy(model) for _ in range(5)]
client_models_bottom_lnf=[copy.deepcopy(model) for _ in range(5)]
client_models_bottom_lmhead=[copy.deepcopy(model) for _ in range(5)]

# Split each client model
for i in range(num_clients):
    client_models_top[i], client_models_bottom[i], client_models_bottom_lnf[i], client_models_bottom_lmhead[i] = split_gpt2_model(model, split_start[i], split_end[i])

client_models_top = [copy.deepcopy(model) for _ in range(num_clients)]
client_models_bottom = [copy.deepcopy(model) for _ in range(num_clients)]
client_models_bottom_lnf=[copy.deepcopy(model) for _ in range(num_clients)]
client_models_bottom_lmhead=[copy.deepcopy(model) for _ in range(num_clients)]

client_model_save_path='./model/'
client_model_save_name='client'
# Split the model for each client
for i in range(num_clients):
    client_models_top[i], client_models_bottom[i], client_models_bottom_lnf[i], client_models_bottom_lmhead[i] = split_gpt2_model(model, split_start[i], split_end[i])
    with open(client_model_save_path+client_model_save_name+str(i)+'_top'+'_'+str(split_start[i])+'_'+str(split_end[i])+'.pkl','wb') as f:
        pickle.dump(client_models_top[i],f)
    with open(client_model_save_path+client_model_save_name+str(i)+'_bottom'+'_'+str(split_start[i])+'_'+str(split_end[i])+'.pkl','wb') as f:
        pickle.dump(client_models_bottom[i],f)
    with open(client_model_save_path+client_model_save_name+str(i)+'_bottom_lnf'+'_'+str(split_start[i])+'_'+str(split_end[i])+'.pkl','wb') as f:
        pickle.dump(client_models_bottom_lnf[i],f)
    with open(client_model_save_path+client_model_save_name+str(i)+'_bottom_lmhead'+'_'+str(split_start[i])+'_'+str(split_end[i])+'.pkl','wb') as f:
        pickle.dump(client_models_bottom_lmhead[i],f)

with open('./embedding/embedding_wte.pkl','wb') as f:
    pickle.dump(model.transformer.wte,f)
with open('./embedding/embedding_wpe.pkl','wb') as f:
    pickle.dump(model.transformer.wpe,f)


with open('./tokenizer/tokenizer.pkl','wb') as f:
    pickle.dump(tokenizer,f)


raise Exception("Datasets and Models Saved!")


# Function to create a full model from client parts and server intermediate layers
def assemble_full_model(client_top, client_bottom, i):
    full_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
    full_model.transformer.h = nn.ModuleList(list(client_top.children()) +
                                             list(model.transformer.h[split_start[i]
                                                                      :split_end[i]]) +
                                             list(client_bottom.children())[:-2])
    full_model.transformer.ln_f = list(client_bottom.children())[-2]
    full_model.lm_head = list(client_bottom.children())[-1]
    print(device)
    return full_model.to(device)


all_losses = []
all_accuracies = []
initial_losses = []
initial_accuracies = []

model = model.to(device)
print(device)
print(model.eval)

data_collator = DataCollatorWithPadding(tokenizer, padding=True)

for client_num in range(num_clients):
    # print(len(tokenized_datasets[client_num]['train']), len(tokenized_datasets[client_num]['test']))
    training_args = Seq2SeqTrainingArguments(
        output_dir=f'./results_{client_num}',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_dir=f'./logs_{client_num}',
        logging_steps=10,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets[client_num]['train'],
        eval_dataset=tokenized_datasets[client_num]['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        #   compute_metrics=compute_metrics
    )

    eval_results = trainer.evaluate()
    print(f"Client {client_num + 1} personalized evaluation results: {eval_results}")
    acc = evaluation(tokenized_datasets[client_num]['test'], model, tokenizer)

    print(acc)
    # 存储初始的 loss 和 accuracy
    initial_losses.append(eval_results['eval_loss'])
    initial_accuracies.append(acc)

all_losses.append(initial_losses)
all_accuracies.append(initial_accuracies)

num_rounds = 10


# Simulate Federated Split Learning with SFT and personalized evaluation
for round in range(num_rounds):  # Number of communication rounds

    round_losses = []
    round_accuracies = []

    for i in range(num_clients):
        # Create DataLoader for each client's split of the dataset
        print_memory_usage()

        torch.cuda.empty_cache()
        client_dataloader = DataLoader(client_datasets[i], sampler=RandomSampler(client_datasets[i]), batch_size=32)

        # Assemble the full model using client's top, server's intermediate, and client's bottom
        full_model = assemble_full_model(client_models_top[i], client_models_bottom[i], i).to(device)
        print("===========================")
        print(round, i)
        print(full_model)
        print_model_size(full_model)
        # Define training arguments and trainer for SFT
        training_args = Seq2SeqTrainingArguments(
            output_dir=f'./results_{i}',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./logs_{round}_{i}',
            logging_steps=10,
            save_strategy="no",
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets[i]['train'],
            eval_dataset=tokenized_datasets[i]['test'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            #   compute_metrics=compute_metrics
        )
        # Perform Soft Fine-Tuning (SFT)
        trainer.train()
        trainer.save_model()

        # After training, update the server's intermediate layers

        # Perform personalized evaluation for the client
        with torch.no_grad():
            eval_results = trainer.evaluate()

        acc = evaluation(tokenized_datasets[i]['test'], full_model, tokenizer)

        # initial_accuracies.append(acc)

        with torch.no_grad():
            # 将 client_models_top 的参数与 GPT-2 的前几层（例如 split_start 定义的范围）同步
            for server_param, client_param in zip(client_models_top[i].parameters(),
                                                  full_model.transformer.h[:split_start[i]].parameters()):
                server_param.data = client_param.data.clone()

        with torch.no_grad():
            # 将 client_models_bottom 的参数与 GPT-2 的后几层（例如 split_end 定义的范围）同步
            for server_param, client_param in zip(client_models_bottom[i].parameters(),
                                                  full_model.transformer.h[split_end[i]:].parameters()):
                server_param.data = client_param.data.clone()
        
        with torch.no_grad():
            for server_param, client_param in zip(
                model.transformer.h[split_start[i]:split_end[i]].parameters(), 
                full_model.transformer.h[split_start[i]:split_end[i]].parameters()):
                server_param.data = client_param.data.clone()

        round_losses.append(eval_results['eval_loss'])
        round_accuracies.append(acc)
        print(f"Client {i + 1} personalized evaluation results: {eval_results}")

    all_losses.append(round_losses)
    all_accuracies.append(round_accuracies)

print(all_losses)
print(all_accuracies)

print("Federated Split Learning with SFT, Weight Updating, and Personalized Evaluation completed.")
