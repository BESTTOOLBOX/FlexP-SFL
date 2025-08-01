import torch


def evaluation(tokenized_data, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):

    print("Check answer tokenization:")
    for ans in ["A","B","C","D"," A"," B"," C"," D"]:
        ids = tokenizer.encode(ans, add_special_tokens=False)
        print(ans, ids, tokenizer.convert_ids_to_tokens(ids))
    model.to(device)
    model.eval()
    print("EVAL")
    # 获取 A/B/C/D 的 token id（根据您的 tokenizer 实际分词结果进行调整）
    valid_answer_tokens = []
    for choice in ["A", "B", "C", "D", " A", " B", " C", " D"]:
        ids = tokenizer.encode(" " + choice, add_special_tokens=False)  # 需与训练时最后答案的格式一致
        if len(ids) == 1:
            valid_answer_tokens.append(ids[0])
        else:
            print(f"Warning: {choice} not a single token. Got {ids}")

    total_correct = 0
    total_instances = 0

    for i, data in enumerate(tokenized_data):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        labels = labels.unsqueeze(0)

        label = labels[:, -1]
        print(i)
        # 检查 label 是否是 A/B/C/D
        if label.item() not in valid_answer_tokens:
            print("??")
            continue

        # 截断输入，预测最后一个token
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        # 不要对attention_mask额外unsqueeze
        attention_mask = attention_mask.bool()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0] # [1, seq_len-1, vocab_size]
            last_logit = logits[:, -1, :]

        predicted_token = torch.argmax(last_logit, dim=-1).item()

        print("Predicted token id:", predicted_token)
        print("Predicted token (text):", tokenizer.decode([predicted_token]))

        print("Label token id:", label.item())
        print("Label token (text):", tokenizer.decode([]))

        # 如果模型输出的并不是 A/B/C/D，则跳过这个样本
        if predicted_token not in valid_answer_tokens:
            continue
        # 假设在获得 predicted_token 和 label 后

        # 此时模型输出与数据标签均是 A/B/C/D 的 token，进行正确性判断
        total_correct += int(predicted_token == label.item())
        total_instances += 1

    accuracy = total_correct / total_instances if total_instances > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}% ({total_correct}/{total_instances})")

    return accuracy
