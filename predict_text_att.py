import os
from PIL import Image
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from tqdm import tqdm
from OCR import load_image, process_image, find_text_and_bounding_box, export_data

def predict_attribute(tokenizer, model, text):
    """ 
    抽出文字の属性を推測する関数
        Args:
            tokenizer: トークナイザー
            model: モデル
            text: 抽出文字を格納したリスト
        
        Returns:
            output (str): 推測した属性
    """
    # プロンプトの記述
    instruction = "書類の項目として、記入欄がどのデータ型にあたるかを選択してください。期間や期限は日付、名前や住所は文字列、経費や金額や個数は数値のように答えてください。"
    input = f"「{text}」という欄がどのデータ型に該当するかを、日付、文字列、数値、単一選択、複数選択の中から最も適切なものを選んでください。"
    
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
    output = output.split("システム: ")[1]
    print(output)
    
    return output

def main():
    input_path = './sample/sample4.jpg'
    # ファイルが存在しない場合、プログラムを終了する
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return

    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # 入力画像の読み込み
    image_original, image_OCR = load_image(input_path)
    
    # 画像処理と領域取得
    image_bw= process_image(image_original)

    # 配列を画像に変換
    image_bw = Image.fromarray(image_bw)
    
    # テキスト抽出とバウンディングボックス検出
    text, bounding_box = find_text_and_bounding_box(image_bw, image_OCR, filename)
    
    # 動作結果をファイルにエクスポート
    results_path = './data/OCR'
    export_data(results_path, text, bounding_box)
    
    # 文字属性の推測
    print('\n starting attributes prediction')
    tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b-chat-gptq")
    model = AutoGPTQForCausalLM.from_quantized("rinna/youri-7b-chat-gptq", use_safetensors=True)
    
    attributes = []
    for i in tqdm(range(len(text))):
        att = predict_attribute(tokenizer, model, text[i])
        attributes.append(att)
        
    for i in range(len(attributes)):
        print(f'att[{i}]: {attributes[i]} ({text[i]})')
        
if __name__ == "__main__":
    main()