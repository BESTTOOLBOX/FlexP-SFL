import torch


def evaluation(tokenized_data, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    评估 GPT-2 模型生成下一个 token 的准确度

    参数:
    - tokenized_data: 已经 tokenized 的数据，包含 input_ids 和 labels。
    - model: GPT-2 模型。
    - device: 设备 (默认为 GPU，如果可用)。

    返回:
    - accuracy: 模型生成的准确率。
    """
    model.to(device)
    model.eval()

    total_correct = 0
    total_instances = 0
    # resx = 0, resy = 0

    for i, data in enumerate(tokenized_data):
        # 提取输入的 input_ids 和真实标签
        input_ids = torch.tensor(data['input_ids']).to(device).unsqueeze(0)

        # remove the last token of input_ids
        label = input_ids[:, -1].item()
        input_ids = input_ids[:, :-1]


        # 前向传播获取模型输出
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # 获取预测的 token
        predicted_token = torch.argmax(logits[0, -1]).item()

        # 计算准确率
        # print("Question: ", tokenizer.decode(input_ids[0]))
        # print("Predicted: ", tokenizer.decode(predicted_token))
        # print("Correct answer: ", tokenizer.decode(label))

        total_correct += int(predicted_token == label)
        total_instances += 1


    # 计算准确率
    print(total_correct, total_instances)
    accuracy = total_correct / total_instances if total_instances > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy
