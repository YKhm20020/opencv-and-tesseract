import torch
from transformers import pipeline

generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

generate_text("整数、文字列、単一選択、複数選択の中で、「氏名」はどれに分類される？")