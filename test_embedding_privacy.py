import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 设置模型路径
model_path = "/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# 输入词
input_text = "hello"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
token_id = input_ids[0, 0].item()

# 获取 embedding 向量
with torch.no_grad():
    input_embedding = model.transformer.wte(input_ids)  # shape: [1, seq_len, hidden_size]
    emb = input_embedding[0, 0]  # shape: [hidden_size]

    print(f"\nOriginal token: '{input_text}' (ID: {token_id})")
    #print("Original embedding vector (truncated):", emb[:5])

    # 创建一个可逆矩阵 A
    hidden_size = emb.shape[0]
    A = torch.randn(hidden_size, hidden_size)
    while torch.linalg.matrix_rank(A) < hidden_size:
        A = torch.randn(hidden_size, hidden_size)  # 保证满秩

    A_inv = torch.inverse(A)

    # ---------- 实验1：过 A -> A^-1 -> lm_head ----------
    emb_transformed1 = A @ emb
    emb_recovered = A_inv @ emb_transformed1
    logits1 = model.lm_head(emb_recovered)
    probs1 = torch.softmax(logits1, dim=-1)
    top1_id_1 = torch.argmax(probs1).item()
    top1_token_1 = tokenizer.decode([top1_id_1])

    print(f"\n[实验1] A -> A^-1 -> lm_head:")
    print("Recovered top-1 token:", repr(top1_token_1))
    print("Probability:", probs1[top1_id_1].item())

    # ---------- 实验2：直接 A -> lm_head ----------
    emb_transformed2 = A @ emb
    logits2 = model.lm_head(emb_transformed2)
    probs2 = torch.softmax(logits2, dim=-1)
    top1_id_2 = torch.argmax(probs2).item()
    top1_token_2 = tokenizer.decode([top1_id_2])

    print(f"\n[实验2] A -> lm_head:")
    print("Transformed top-1 token:", repr(top1_token_2))
    print("Probability:", probs2[top1_id_2].item())
