import torch
from tqdm import tqdm
from data_utils import get_dataloader, tokenizer

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