import os
from typing import List, Tuple
from PIL import Image
import numpy as np
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from tqdm import tqdm
from OCR import load_image, process_image, find_text_and_bounding_box, export_data

def extract_text(image_path: str, filename: str) -> Tuple[List[str], List[np.ndarray]]:
    """ 抽出文字とバウンディングボックスの座標を検出する関数
    
    画像読み取りから文字とバウンディングボックス座標の抽出までを実行する関数
    
        Args:
            image_path (str): 入力画像のパス
            filename (str): ファイルの名前
        
        Returns:
            Tuple[List[str], List[numpy.ndarray]]: 抽出した文字、抽出文字を囲うバウンディングボックス
            
        Note:
            txt (list[str]): 抽出した文字
            b_box (list[numpy.ndarray]): 抽出文字を囲うバウンディングボックス
    
    """
    # 入力画像の読み込み
    image_original, image_OCR = load_image(image_path)
    
    # 画像処理と領域取得
    image_bw = process_image(image_original)

    # 配列を画像に変換
    image_bw = Image.fromarray(image_bw)
    
    # テキスト抽出とバウンディングボックス検出
    txt, b_box = find_text_and_bounding_box(image_bw, image_OCR, filename)
    
    # 動作結果をファイルにエクスポート
    results_path = './data/OCR'
    export_data(results_path, txt, b_box)
    
    return txt, b_box


def predict_text_attribute(tokenizer, model, txt: str) -> str:
    """ 属性を推測する関数
    
    文字抽出によって抽出した文字の属性を推測する関数
    
        Args:
            tokenizer: トークナイザー
            model: モデル
            txt (str): 抽出した文字
        
        Returns:
            att (str): 推測した属性
            
        Note:
            tokenizer: <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'> 
            model: <class 'auto_gptq.modeling.llama.LlamaGPTQForCausalLM'>
    
    """
    # プロンプトの記述
    instruction = "書類の項目として、記入欄がどのデータ型にあたるかを選択してください。期間や期限は日付、名前や住所は文字列、経費や金額や個数は数値のように答えてください。"
    input = f"「{txt}」という欄がどのデータ型に該当するかを、日付、文字列、数値、単一選択、複数選択の中から最も適切なものを選んでください。"
    
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

    att = tokenizer.decode(output_ids.tolist()[0])
    att = att.replace("</s>", "")
    att = att.split("システム: ")[1]
    print(att)
    
    return att
    
    
def link_attribute_to_text(tokenizer, model, txts: str) -> List[str]:
    """ 属性を推測する関数
    
    文字抽出によって抽出した文字の属性を推測する関数
    
        Args:
            tokenizer: トークナイザー
            model: モデル
            txts (str): 抽出した文字
        
        Returns:
            atts (List[str]): 推測した属性
            
        Note:
            tokenizer: <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'> 
            model: <class 'auto_gptq.modeling.llama.LlamaGPTQForCausalLM'>
    
    """
    atts = []
    for i in tqdm(range(len(txts))):
        att = predict_text_attribute(tokenizer, model, txts[i])
        atts.append(att)
        
    for i in range(len(atts)):
        print(f'att[{i}]: {atts[i]} ({txts[i]})')
    
    return atts


def main():
    input_path = './sample/sample4.jpg'
    # ファイルが存在しない場合、プログラムを終了する
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return

    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b-chat-gptq")
    model = AutoGPTQForCausalLM.from_quantized("rinna/youri-7b-chat-gptq", use_safetensors=True)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    texts, bounding_boxes = extract_text(input_path, filename)
    
    # 文字属性の推測
    print('\nstarting attributes prediction')
    text_attributes = link_attribute_to_text(tokenizer, model, texts)
    


if __name__ == "__main__":
    main()