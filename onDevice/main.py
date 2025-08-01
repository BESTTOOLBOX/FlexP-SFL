import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, random_split
import random
import copy
import torch.nn as nn
import datasets
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device setup complete:", device)
print("Available GPUs:", torch.cuda.device_count())

# Load the Flan-T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base')
tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-base')
tokenizer.pad_token = tokenizer.eos_token

# Define the dataset loading function for MMLU
# Load the dataset for multiple clients (abstract_algebra, anatomy, astronomy, business_ethics, clinical_knowledge)
mmlu_tasks = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge']

# Load dataset using the format you've specified
from datasets import concatenate_datasets, load_dataset

client_datasets = [load_dataset("cais/mmlu", task) for task in mmlu_tasks]

merged_datasets = []
for dataset in client_datasets:
    test_dataset = dataset['test']
    validation_dataset = dataset['validation']
    dev_dataset = dataset['dev']

    merged_dataset = concatenate_datasets([test_dataset, validation_dataset, dev_dataset])
    merged_datasets.append(merged_dataset)

train_test_split_datasets = []
for dataset in merged_datasets:
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_test_split_datasets.append({"train": split_dataset["train"], "test": split_dataset["test"]})


client_datasets = train_test_split_datasets


# Preprocessing function for tokenizing the question and choices
def preprocess_mmlu(examples):
    questions = examples['question']
    choices_list = examples['choices']
    answers = examples['answer']
    
    inputs = []
    attention_masks = []
    labels = []
    decoder_inputs = []
    
    for question, choices, answer in zip(questions, choices_list, answers):
        # Ensure each choice is a string and flatten the list if needed
        flattened_choices = [str(choice) for choice in choices]
        
        # 构造输入提示，删除多余的方括号
        prompt = f"Question: {question}. Choices: A) {choices[0]}, B) {choices[1]}, C) {choices[2]}, D) {choices[3]}. Output a single letter."
        
        # Tokenize the prompt (input to the encoder) and add padding/truncation
        tokenized_input = tokenizer(prompt, max_length=1024, padding='max_length', truncation=True)
        
        # Create the target (correct answer) as a sequence, e.g., "Answer: A"
        answer_text = f"{chr(65 + int(answer))}"  # Convert index to corresponding letter
        tokenized_answer = tokenizer(answer_text, max_length=10, padding='max_length', truncation=True)

        # Append input_ids, attention_mask, and tokenized labels (as sequences)
        inputs.append(tokenized_input['input_ids'])
        attention_masks.append(tokenized_input['attention_mask'])
        labels.append(tokenized_answer['input_ids'])  # Use tokenized answer as labels
        
        # Append decoder input_ids for sequence-to-sequence training
     #   decoder_inputs.append(tokenized_answer['input_ids'])
    # Return tokenized inputs, attention_mask, labels, and decoder_input_ids as lists (required for batched=True)
    return {
        'input_ids': inputs,
        'attention_mask': attention_masks,
        'labels': labels,  # Ensure the labels are sequences matching the decoder outputs
     #   'decoder_input_ids': decoder_inputs
    }

# Preprocess the datasets
# 对每个数据集的 "train" 和 "test" 部分分别应用 .map()
tokenized_datasets = []
for dataset in client_datasets:
    if isinstance(dataset["train"], str):
        print("Error: 'train' is a string, expected Dataset object.")
    else:
        tokenized_train = dataset["train"].map(preprocess_mmlu, batched=True)
        tokenized_test = dataset["test"].map(preprocess_mmlu, batched=True)
        tokenized_datasets.append({"train": tokenized_train, "test": tokenized_test})

# Adjust number of clients
num_clients = 5

# Split the model into three parts: client_top, server_intermediate, client_bottom
def split_gpt2_model(model, split_encoder_end, split_decoder_start):
    client_top = nn.Sequential(*list(model.encoder.block[:split_encoder_end]))
    client_bottom = nn.Sequential(*list(model.decoder.block[split_decoder_start:]), model.lm_head)
    return client_top, client_bottom

# Define the split points for each client
split_start = [3, 3, 9, 6, 6]
split_end = [10, 10, 10, 10, 10]

client_models_top = [copy.deepcopy(model) for _ in range(num_clients)]
client_models_bottom = [copy.deepcopy(model) for _ in range(num_clients)]

# Split the model for each client
for i in range(num_clients):
    client_models_top[i], client_models_bottom[i] = split_gpt2_model(model, split_start[i], split_end[i])

# Function to create a full model from client parts and server intermediate layers
def assemble_full_model(client_top, client_bottom, i):
    full_model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base')
    full_model.encoder.block = nn.ModuleList(list(client_top.children()) + 
                                             list(model.encoder.block[split_start[i]:]))
    full_model.decoder.block = nn.ModuleList(list(model.decoder.block[:split_end[i]]) + 
                                             list(client_bottom.children())[:-1])
    full_model.lm_head = list(client_bottom.children())[-1]
    return full_model.to(device)

all_losses = []
all_accuracies = []
initial_losses = []
initial_accuracies = []

import numpy as np

from scipy.special import softmax

def compute_metrics(p):
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # 对logits进行softmax转换成概率分布
    probabilities = softmax(predictions, axis=-1)
    # 获取预测类别
    preds = np.argmax(probabilities, axis=-1)
    
    # 确保 preds 和 label_ids 的形状一致
    if preds.shape != p.label_ids.shape:
        raise ValueError(f"Predictions shape {preds.shape} and labels shape {p.label_ids.shape} don't match!")
    
    # 计算准确率
    correct_predictions = np.all(preds == p.label_ids, axis=-1)  # 每行都完全匹配才为 True
    
    # 计算每行的准确性
    accuracy = np.mean(correct_predictions)  # 这里 correct_predictions 是一个布尔数组
    
    return {'accuracy': accuracy}



for client_num in range(num_clients):
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
        train_dataset=tokenized_datasets[client_num]['test'],
        eval_dataset=tokenized_datasets[client_num]['test'], 
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    eval_results = trainer.evaluate()
    print(f"Client {client_num+1} personalized evaluation results: {eval_results}")
    
    # Store initial loss and accuracy
    initial_losses.append(eval_results['eval_loss'])
    initial_accuracies.append(eval_results.get('eval_accuracy', None))
    
all_losses.append(initial_losses)
all_accuracies.append(initial_accuracies)

num_rounds = 10



# Simulate Federated Split Learning with SFT and personalized evaluation
for round in range(num_rounds):  # Number of communication rounds
    
    round_losses = []
    round_accuracies = []

    for i in range(num_clients):
       # print_memory_usage()

        torch.cuda.empty_cache()
        client_dataloader = DataLoader(tokenized_datasets[i]['test'], sampler=RandomSampler(tokenized_datasets[i]['test']), batch_size=32)

        full_model = assemble_full_model(client_models_top[i], client_models_bottom[i], i).to(device)
        print("===========================")
        print(round, i)
        print(full_model)
       # print_model_size(full_model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=f'./results_{i}',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./logs_{i}',
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
        )
        trainer = Seq2SeqTrainer(
            model=full_model,
            args=training_args,
            train_dataset=tokenized_datasets[i]['train'],
            eval_dataset=tokenized_datasets[i]['test'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        # Perform Soft Fine-Tuning (SFT)
        trainer.train()

        # Update the server's intermediate layers
        # Update the server's intermediate layers

        # Personalized evaluation
        eval_results = trainer.evaluate()

        with torch.no_grad():
            for server_param, client_param in zip(client_models_top[i].parameters(), full_model.encoder.block[:split_start[i]].parameters()):
                server_param.data = client_param.data.clone()
        with torch.no_grad():
            for server_param, client_param in zip(client_models_bottom[i].parameters(), full_model.decoder.block[split_end[i]:].parameters()):
                server_param.data = client_param.data.clone()
        with torch.no_grad():
            for server_param, client_param in zip(model.encoder.block[split_start[i]:].parameters(), full_model.encoder.block[split_start[i]:].parameters()):
                server_param.data = client_param.data.clone()
        with torch.no_grad():
            for server_param, client_param in zip(model.decoder.block[:split_end[i]].parameters(), full_model.decoder.block[:split_end[i]].parameters()):
                server_param.data = client_param.data.clone()

        
        round_losses.append(eval_results['eval_loss'])
        round_accuracies.append(eval_results.get('eval_accuracy', None))
        print(f"Client {i+1} personalized evaluation results: {eval_results}")
        
    all_losses.append(round_losses)
    all_accuracies.append(round_accuracies)
    
print(all_losses)
print(all_accuracies)
print("Federated Split Learning with SFT, Weight Updating, and Personalized Evaluation completed.")
