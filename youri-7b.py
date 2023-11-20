import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b-chat-gptq")
model = AutoGPTQForCausalLM.from_quantized("rinna/youri-7b-chat-gptq", use_safetensors=True)

instruction = "書類の項目として、記入欄がどのデータ型にあたるかを選択してください。"
input = "「個数」という欄がどのデータ型に該当するかを、日付、文字列、数値、単一選択、複数選択の中から選んでください。"

context = [
    {
        "speaker": "設定",
        "text": instruction
    },
    {
        "speaker": "ユーザー",
        "text": input
    }
]
prompt = [
    f"{uttr['speaker']}: {uttr['text']}"
    for uttr in context
]
prompt = "\n".join(prompt)
prompt = (
    prompt
    + "\n"
    + "システム: "
)
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        input_ids=token_ids.to(model.device),
        max_new_tokens=100,
        do_sample=True,
        temperature=0.5,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0])
output = output.replace("</s>", "")
print(output)