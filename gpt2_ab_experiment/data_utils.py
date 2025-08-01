from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained("/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/")
tokenizer.pad_token = tokenizer.eos_token

def get_dataloader(split_idx=0, total_parts=2, num_samples=5):
    dataset = load_dataset(
        "parquet", 
        data_files="/new_disk/houyz/gjx_SplitFM/mmlu_dataset/professional_psychology/test-00000-of-00001.parquet"
    )["train"].select(range(num_samples))

    part = [i for i in range(len(dataset)) if i % total_parts == split_idx]
    subset = dataset.select(part)

    def tokenize(batch):
        print(batch)
        prompts = [
            f"Question: {q} Choices: {', '.join(c)} Answer:" 
            for q, c in zip(batch["question"], batch["choices"])
        ]
        targets = [batch["choices"][i][a] for i, a in enumerate(batch["answer"])]
        full_texts = [p + " " + t for p, t in zip(prompts, targets)]
        #print(full_texts)
        raise Exception("Over")
        result = tokenizer(full_texts, padding="max_length", truncation=True, max_length=64)
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = subset.map(tokenize, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return DataLoader(tokenized, batch_size=4)