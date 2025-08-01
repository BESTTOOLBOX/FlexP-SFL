from transformers import GPT2LMHeadModel, AdamW
import torch
from tqdm import tqdm
from data_utils import get_dataloader, tokenizer

def train_plain():
    model = GPT2LMHeadModel.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/").cuda()
    model.resize_token_embeddings(len(tokenizer))
    optimizer = AdamW(model.parameters(), lr=5e-5)
    dataloader = get_dataloader(split_idx=0, total_parts=1, num_samples=500)

    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[PLAIN] Epoch {epoch+1} loss: {total_loss / len(dataloader):.4f}")

    return model