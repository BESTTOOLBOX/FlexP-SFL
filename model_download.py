import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

model_name = "openai-community/gpt2"

while True:
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir_use_symlinks=False,
            #ignore_patterns=["*.bin"],
            local_dir=model_name,
            token="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",
            resume_download=True
        )
        break
    except:
        pass
