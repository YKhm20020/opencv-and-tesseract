from llama_cpp import Llama

# プロンプトを記入
prompt = """Which of the following is the label of "年月日" in Japanese: date, integer, string, single selection or multiple selection? Choose only these 5 words."
Assistant:"""

# ダウンロードしたModelをセット
llm = Llama(model_path="./Llama/models/llama-2-7b-chat.ggmlv3.q8_0.bin")

# 生成実行
output = llm(
    prompt,max_tokens=500,stop=["System:", "User:", "Assistant:"],echo=True,
)

print(output)