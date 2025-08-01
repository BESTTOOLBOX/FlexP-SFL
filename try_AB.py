import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = GPT2LMHeadModel.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/")
tokenizer.pad_token = tokenizer.eos_token


def get_dataloader(split_idx=0, total_parts=2, num_samples=100):
    dataset_dict = load_dataset("parquet", data_files="/new_disk/houyz/gjx_SplitFM/mmlu_dataset/professional_psychology/test-00000-of-00001.parquet")
    dataset = dataset_dict["train"].select(range(num_samples))

    part = [i for i in range(len(dataset)) if i % total_parts == split_idx]
    subset = dataset.select(part)

    def tokenize(example):
        choices_str = " ".join([f"{chr(65+i)}. {c}" for i, c in enumerate(example["choices"])])
        prompt = f"{example['question']} {choices_str} Answer:"
        target_letter = chr(65 + example["answer"])
        encoded = tokenizer(prompt + " " + target_letter, padding="max_length", truncation=True, max_length=64)
        encoded["labels"] = encoded["input_ids"]
        return encoded

    tokenized = subset.map(tokenize)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return DataLoader(tokenized, batch_size=4)


def random_invertible_matrix(dim):
    A = torch.randn(dim, dim)
    while torch.linalg.matrix_rank(A) < dim:
        A = torch.randn(dim, dim)
    return A.to(device), torch.inverse(A).to(device)


def train_normal(model, dataloader, epochs=1):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(epochs):
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            labels = torch.tensor(batch["labels"]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model


def train_with_matrix(model, dataloader, M, epochs=1):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(epochs):
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            embeds = model.transformer.wte(input_ids)
            embeds = embeds @ M.T  # Apply matrix M

            hidden_states = model.transformer(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state

            shifted_labels = labels[:, 1:]
            shifted_hidden = hidden_states[:, :-1, :]

            target_embed = model.transformer.wte(shifted_labels) @ M.T
            loss = nn.MSELoss()(shifted_hidden, target_embed)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model


def evaluate(model, dataloader, M=None, M_inv=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            embeds = model.transformer.wte(input_ids)
            if M is not None:
                embeds = embeds @ M.T

            hidden = model.transformer(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
            if M_inv is not None:
                hidden = hidden @ M_inv.T

            logits = model.lm_head(hidden)
            preds = logits.argmax(dim=-1)

            correct += (preds[:, 1] == input_ids[:, 1]).sum().item()
            total += input_ids.size(0)
    return correct / total
    
def evaluate(model):
    model.eval()
    dataloader = get_dataloader(split_idx=0, num_samples=100)

    correct = 0
    total = 0
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

            last_token_indices = (attention_mask.sum(dim=1) - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, logits.size(-1))
            last_logits = logits.gather(1, last_token_indices).squeeze(1)

            preds = torch.argmax(last_logits, dim=-1)
            decoded_preds = [tokenizer.decode([p]).strip().upper() for p in preds.cpu()]

            for i in range(len(decoded_preds)):
                label_token_id = labels[i][attention_mask[i] == 1][-1]
                true_answer = tokenizer.decode([label_token_id.item()]).strip().upper()
                if decoded_preds[i] == true_answer:
                    correct += 1
                total += 1

    print(f"[EVAL] Accuracy: {correct}/{total} = {correct / total:.4f}")


if __name__ == "__main__":
    dim = model.config.hidden_size
    A, A_inv = random_invertible_matrix(dim)
    B, B_inv = random_invertible_matrix(dim)


    dataloader_A = get_dataloader(split_idx=0)
    dataloader_B = get_dataloader(split_idx=1)


    model1 = GPT2LMHeadModel.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/").to(device)
    model1 = train_normal(model1, dataloader_A)
    acc1 = evaluate(model1, dataloader_A)
    print(f"[Experiment 1] normal finetune accuracy: {acc1:.4f}")


    model2 = GPT2LMHeadModel.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/").to(device)
    model2 = train_with_matrix(model2, dataloader_A, A)
    model2 = train_with_matrix(model2, dataloader_B, B)
    acc2_A = evaluate(model2, dataloader_A, A, A_inv)
    acc2_B = evaluate(model2, dataloader_B, B, B_inv)
    print(f"[Experiment 2] accuracy with A: {acc2_A:.4f}")
    print(f"[Experiment 2] accuracy with B: {acc2_B:.4f}")
