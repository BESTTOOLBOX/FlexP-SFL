import torch

def evaluation(tokenized_data, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    评估 GPT-2 模型生成下一个 token 的准确度，仅统计预测结果为 A/B/C/D 的样本。
    """
    model.to(device)
    model.eval()

    # 假设 "A","B","C","D" 对应的单独 token 为32,33,34,35
    # 如果不一致请根据实际情况更改
    valid_answer_tokens = [317, 327, 347, 360]

    total_correct = 0
    total_instances = 0

    for i, data in enumerate(tokenized_data):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        input_ids = input_ids.unsqueeze(0)        # [1, seq_len]
        attention_mask = attention_mask.unsqueeze(0)  # [1, seq_len]
        labels = labels.unsqueeze(0)             # [1, seq_len]

        # 这里假设只最后一个token是有效label
        input_ids = input_ids[:, :-1]         # 去掉最后一个token作为输入
        attention_mask = attention_mask[:, :-1]

        # 转换 attention_mask 为 bool
        attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)

        label = labels[:, -1]  # 最后一个 token 的 label
        label_id = label.item()

        # 前向传播
        with torch.no_grad():
            outputs, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            # outputs: [1, seq_len-1, vocab_size]
            last_logit = outputs[:, -1, :]  # [1, vocab_size]

        predicted_token = torch.argmax(last_logit, dim=-1).item()

        # 如果预测的不是 A/B/C/D，则跳过这个样本
        if predicted_token not in valid_answer_tokens:
            continue

        # 如果您还想确保 label 也是 A/B/C/D，可以加上下面这行检查：
        # if label_id not in valid_answer_tokens:
        #     continue

        # 若预测在 A/B/C/D 中，将其计入统计
        total_correct += int(predicted_token == label_id)
        total_instances += 1

    accuracy = total_correct / total_instances if total_instances > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}% ({total_correct}/{total_instances})")

    return accuracy

import torch

def evaluation2(tokenized_data, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    评估 GPT-2 模型生成下一个 token 的准确度

    参数:
    - tokenized_data: 已经 tokenized 的数据，支持索引访问的形式，如列表或Dataset，包含 input_ids、attention_mask 和 labels。
    - model: GPT-2 模型。
    - tokenizer: 用于解码预测的 tokenizer。
    - device: 设备 (默认为 GPU，如果可用)。

    返回:
    - accuracy: 模型生成的准确率。
    """
    model.to(device)
    model.eval()

    total_correct = 0
    total_instances = 0

    for i, data in enumerate(tokenized_data):
        # data 应该是一个包含 input_ids, attention_mask, labels 的字典
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        # 扩展成 batch 维度：因为full_model是按batch处理的
        input_ids = input_ids.unsqueeze(0)        # [1, seq_len]
        attention_mask = attention_mask.unsqueeze(0)  # [1, seq_len]
        labels = labels.unsqueeze(0)             # [1, seq_len]

        # 去掉最后一个 token 来预测下一个 token
        # 假设要预测最后一个token
        # 注意: 根据您的预处理，只最后一个token是有效label，其余为-100
        # 因此这里如果要评估最后一个 token 的预测情况，可以直接使用最后一个 token 的位置。
        # 也可以像之前那样移除最后一个token，再预测最后一个.
        # 这里假设只预测最后一个 token (labels中非 -100 的那个)
        
        # 首先找到最后一个非 -100 的位置(理论上就是最后一个token)
        # 但由于您的预处理是只最后一个是有效label，可以直接使用最后一个 token 的label
        # 将 attention_mask 与 input_ids 切掉最后一个token，因为要用前面输入预测最后的token
        input_ids = input_ids[:, :-1]         # 去掉最后一个token作为输入
        attention_mask = attention_mask[:, :-1]

        # 将 attention_mask 转换为 bool 类型以满足 scaled_dot_product_attention 要求
        attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
        # 现在 attention_mask 形状：[1,1,1,seq_len-1]

        # labels中只要最后一个token的label进行对比
        label = labels[:, -1]  # 最后一个 token 的 label

        # 前向传播获取模型输出
        with torch.no_grad():
            outputs, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            # outputs 是最终logits: [1, seq_len-1, vocab_size]
            # 我们要预测最后一个token，所以看 outputs 的最后一个位置的logits
            last_logit = outputs[:, -1, :]  # [1, vocab_size]

        # 获取预测的 token
        predicted_token = torch.argmax(last_logit, dim=-1).item()
        print(label.item(), predicted_token)
        # 计算准确率
        total_correct += int(predicted_token == label.item())
        total_instances += 1

    # 计算并打印准确率
    accuracy = total_correct / total_instances if total_instances > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}% ({total_correct}/{total_instances})")

    return accuracy