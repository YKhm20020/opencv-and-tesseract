# 使用したプロンプトフォーマット
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


import torch
import transformers
from transformers import AutoTokenizer
name = 'Jumtra/mpt-7b-inst'
config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'torch'
config.init_device = 'cuda:0' # For fast initialization directly on GPU!
tokenizer = AutoTokenizer.from_pretrained(
    "Jumtra/mpt-7b-inst"
)
model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
).to("cuda:0")
model.eval()

input_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction = "ニューラルネットワークとは何ですか？")

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
input_length = inputs.input_ids.shape[1]

# Without streaming
with torch.no_grad():
    generation_output = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.01,
        top_p=0.01,
        top_k=60,
        repetition_penalty=1.1,
        return_dict_in_generate=True,
        remove_invalid_values=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
token = generation_output.sequences[0, input_length:]
output = tokenizer.decode(token)
print(output)

#ニューラルネットワーク（NN）は、人工知能の分野で使用される深い学習アルゴリズムの一種です。これらのアルゴリズムは、データを使って自動的に学習し、特定の目的を達成するために予測や決定を行うことができます。ニューラルネットワークは、多くの異なるアプリケーションで使用されており、自動車の運転システム、検索エンジン、画像認識などです。<|endoftext|>
