from llama_cpp import Llama

# プロンプトを記入
prompt = """"Which of the following is the label of "年齢" in Japanese ? 'Answer only in date, number, string, single selection or multiple selection.",
Assistant:"""

# ダウンロードしたModelをセット
#llm = Llama(model_path="./Llama/models/nsql-llama-2-7b.Q4_K_M.gguf", n_gpu_layers=40)
llm = Llama(model_path="./Llama/models/nsql-llama-2-7b.Q5_K_M.gguf", n_gpu_layers=40)

# 生成実行
output = llm(
    prompt,max_tokens=500,stop=["System:", "User:", "Assistant:"],echo=True,
)

# 生成されたテキストを抽出
generated_text = output['choices'][0]['text']

# "Assistant:" 以降のテキストを取得
assistant_response = generated_text.split("Assistant:")[1].strip()

# 結果を出力
print("Assistant's Response:")
print(assistant_response)

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-llama-2-7B")
# model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-llama-2-7B", torch_dtype=torch.bfloat16)

# text = """CREATE TABLE stadium (
#     stadium_id number,
#     location text,
#     name text,
#     capacity number,
#     highest number,
#     lowest number,
#     average number
# )

# CREATE TABLE singer (
#     singer_id number,
#     name text,
#     country text,
#     song_name text,
#     song_release_year text,
#     age number,
#     is_male others
# )

# CREATE TABLE concert (
#     concert_id number,
#     concert_name text,
#     theme text,
#     stadium_id text,
#     year text
# )

# CREATE TABLE singer_in_concert (
#     concert_id number,
#     singer_id text
# )

# -- Using valid SQLite, answer the following questions for the tables provided above.

# -- What is the maximum, the average, and the minimum capacity of stadiums ?

# SELECT"""

# input_ids = tokenizer(text, return_tensors="pt").input_ids

# generated_ids = model.generate(input_ids, max_length=500)
# print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))