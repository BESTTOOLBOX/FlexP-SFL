import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 设定模型路径
model_path = "/new_disk/houyz/gjx_SplitFM/openai-community/gpt2/"

# 加载 tokenizer 和模型
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()  # 评估模式

# 选择一个词语作为输入
input_text = "hello"
input_ids = tokenizer.encode(input_text, return_tensors='pt')  # shape: [1, seq_len]
token_id = input_ids[0, 0].item()

# 获取 input embedding 层的向量
with torch.no_grad():
    input_embedding = model.transformer.wte(input_ids)  # shape: [1, seq_len, hidden_size]
    print(f"Input token: '{input_text}' (ID: {token_id})")
    print("Input embedding vector (first token):", input_embedding[0, 0])

    # 将该 embedding 直接过输出层（lm_head）
    output_logits = model.lm_head(input_embedding)  # shape: [1, seq_len, vocab_size]

    # 只看第一个 token 的结果
    logits = output_logits[0, 0]
    probs = torch.softmax(logits, dim=-1)

    top1_id = torch.argmax(probs).item()
    top1_token = tokenizer.decode([top1_id])
    print("\nTop-1 predicted token after lm_head + softmax:", repr(top1_token))
    print("Probability:", probs[top1_id].item())

