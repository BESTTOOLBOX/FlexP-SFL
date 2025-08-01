from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch
import torch.nn as nn
from tqdm import tqdm
from data_utils import get_dataloader, tokenizer
import os
import pickle

def random_invertible_matrix(dim, device):
    A = torch.randn(dim, dim)
    while torch.linalg.matrix_rank(A) < dim:
        A = torch.randn(dim, dim)
    return A.to(device), torch.inverse(A).to(device)
    
def save_matrix(path, matrix, matrix_inv):
    with open(path, "wb") as f:
        pickle.dump({"matrix": matrix, "matrix_inv": matrix_inv}, f)

def load_matrix(path, device):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["matrix"].to(device), data["matrix_inv"].to(device)

def train_with_matrix(model, dataloader, M, epochs, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
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
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"[MATRIX] Epoch {epoch+1} loss: {total_loss / len(dataloader):.4f}")
    return model


def evaluate(model, dataloader, M=None, M_inv=None):
    model.eval()
    dataloader = get_dataloader(split_idx=0, total_parts=1, num_samples=100)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            embeds = model.transformer.wte(input_ids)
            if M is not None:
                embeds = embeds @ M.T

            hidden = model.transformer(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
            if M_inv is not None:
                hidden = hidden @ M_inv.T

            logits = model.lm_head(hidden)
            last_token_indices = (attention_mask.sum(dim=1) - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, logits.size(-1))
            last_logits = logits.gather(1, last_token_indices).squeeze(1)

            preds = torch.argmax(last_logits, dim=-1)
            decoded_preds = [tokenizer.decode([p]).strip().upper() for p in preds.cpu()]

            for i in range(len(decoded_preds)):
                label_token_id = labels[i][attention_mask[i] == 1][-1]
                true_answer = tokenizer.decode([label_token_id.item()]).strip().upper()

                print(f"[SAMPLE {total + 1}] Predicted: {decoded_preds[i]} | True: {true_answer}")

                if true_answer in decoded_preds[i]:
                    correct += 1
                total += 1

    print(f"[MATRIX] Accuracy: {correct}/{total} = {correct / total:.4f}")
    return correct / total


def train_ab():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2 = GPT2LMHeadModel.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/").to(device)
    print("Model Ready!")
    tokenizer = GPT2Tokenizer.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/")
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer Ready!")
    dim = model2.config.hidden_size
    os.makedirs("matrices", exist_ok=True)

    matrix_A_path = "matrices/matrix_A.pkl"
    matrix_B_path = "matrices/matrix_B.pkl"

    if os.path.exists(matrix_A_path):
        A, A_inv = load_matrix(matrix_A_path, device)
        print("Loaded matrix A from pickle.")
    else:
        A, A_inv = random_invertible_matrix(dim, device)
        save_matrix(matrix_A_path, A, A_inv)
        print("Generated and saved matrix A.")

    if os.path.exists(matrix_B_path):
        B, B_inv = load_matrix(matrix_B_path, device)
        print("Loaded matrix B from pickle.")
    else:
        B, B_inv = random_invertible_matrix(dim, device)
        save_matrix(matrix_B_path, B, B_inv)
        print("Generated and saved matrix B.")
    print("Matrix Ready!")
    dataloader_A = get_dataloader(split_idx=0, total_parts=2)
    dataloader_B = get_dataloader(split_idx=1, total_parts=2)
    print("Data Ready!")
    model2 = train_with_matrix(model2, dataloader_A, A, 2, device)
    model2 = train_with_matrix(model2, dataloader_B, B, 2, device)
    acc2_A = evaluate(model2, dataloader_A, A, A_inv)
    acc2_B = evaluate(model2, dataloader_B, B, B_inv)
    print(f"[Experiment 2] accuracy with A: {acc2_A:.4f}")
    print(f"[Experiment 2] accuracy with B: {acc2_B:.4f}")
    return model2