import os
import sys
from typing import List
import cv2
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from tqdm import tqdm
from OCR import create_OCR_directories, load_OCR_image, process_image_OCR, find_text_and_bounding_box, export_OCR_data


def predict_text_attribute(tokenizer, model, txt: str) -> str:
    """ 属性を推測する関数
    
    文字抽出によって抽出した文字の属性を推測する関数
    
        Args:
            tokenizer: トークナイザー
            model: モデル
            txt (str): 抽出した文字
        
        Returns:
            str: 推測した属性
            
        Note:
            tokenizer: <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'> 
            model: <class 'auto_gptq.modeling.llama.LlamaGPTQForCausalLM'>
            
            返答から抽出文字を削除する際、最初のひとつ目に限定することで、
            データ型として判定した結果が変換されることを防ぐ。
            傾向として、抽出文字が返答に含まれるのは、出力の先頭であることが多い。
    
    """
    
    # 属性が明らかな場合は、推測前に抽出文字をそのまま属性として返す。
    date_candidate = ['年', '月', '日', '年月日', '生年月日', '期間', '期限']
    number_candidate = ['個数', '金額', '単価', 'TEL', 'FAX']
    string_candidate = ['住所', '氏名']
    
    if txt == '日付' or txt in date_candidate:
        return 'date'
    elif txt == '数値' or txt in number_candidate:
        return 'number'
    elif txt == '文字列' or txt in string_candidate:
        return 'string'
    
    # プロンプトの記述
    instruction = """
    # 命令書:
    以下の制約条件にあてはまるものを出力せよ。
    
    # 制約条件
    ・記入欄に記入する内容が、日付、文字列、数値の中から、どのデータ型が最も適切であるかを選択する。
    ・出力は短く、あてはまるデータ型のみとする。
    ・例として、年月日などは日付、氏名などは文字列、金額などは数値があてはまる。
    """
    input = f"「{txt}」という欄は、どのデータ型に該当するか。"
    
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
            max_new_tokens=50,
            do_sample=True,
            temperature=0.6,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0])
    output = output.replace("</s>", "")
    output = output.split("システム: ")[1]
    print(output)
    
    # 判定属性の補正
    # 返答から抽出文字を削除（最初のひとつ目に限定）
    output.replace(txt, '', 1)
    
    if '日' in output:
        att = 'date'
    elif '数' in output:
        att = 'number'
    else:
        att = 'string'
        
    print(att)
        
    return att
    
    
def link_attribute_to_text(tokenizer, model, txts: str) -> List[str]:
    """ 抽出文字と推測属性を紐づけする関数
    
    各抽出結果に対して、推測した属性を紐付ける関数
    
        Args:
            tokenizer: トークナイザー
            model: モデル
            txts (str): 抽出した文字
        
        Returns:
            List[str]: 全抽出文字について推測した属性のリスト
            
        Note:
            tokenizer: <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'> 
            model: <class 'auto_gptq.modeling.llama.LlamaGPTQForCausalLM'>
    
    """
    att_with_text = [predict_text_attribute(tokenizer, model, txt) for txt in tqdm(txts)]

    for i in range(len(att_with_text)):
        print(f'att[{i}]: {att_with_text[i]} ({txts[i]})')

    return att_with_text


def main():
    # ディレクトリ作成
    create_OCR_directories()
    
    try:
        #input_path = './sample/sample4.jpg'
        #input_path = './sample/sample.jpg'
        input_path = './sample/seikyuu.jpg'
        
        #input_path =  './sample/P/3．入出退健康管理簿.pdf'
        #input_path =  './sample/P/13-3-18 入出退健康管理簿（確認印欄あり）.pdf'
        #input_path =  './sample/P/20230826_富士瓦斯資料_設備保安点検01.pdf'
        
        # ファイルが存在しない場合の例外処理
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"The file '{input_path}' does not exist.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit()

    file_name = os.path.splitext(os.path.basename(input_path))[0]
    
    tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b-chat-gptq")
    model = AutoGPTQForCausalLM.from_quantized("rinna/youri-7b-chat-gptq", use_safetensors=True)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # 入力画像の読み込み
    image_original, image_OCR = load_OCR_image(input_path)
    
    # 画像処理
    image_bw = process_image_OCR(image_original)
    
    # テキスト抽出とバウンディングボックス検出
    text, bounding_box = find_text_and_bounding_box(image_bw, image_OCR, file_name)
    
    # 動作結果をファイルにエクスポート
    export_OCR_data(text, bounding_box, file_name)
    
    # 文字属性の推測
    print('\nstarting attributes prediction')
    
    text_attributes = link_attribute_to_text(tokenizer, model, text)
    
    # 画像への描画
    for i in range(len(text)):
        print(f'string[{i}] {bounding_box[i]} : {text[i]}') # 座標と文字列を出力
        cv2.rectangle(image_OCR, bounding_box[i][0], bounding_box[i][1], (0, 0, 255), 1) # 検出した箇所を赤枠で囲む
        cv2.putText(image_OCR, f'{str(i)}: {text_attributes[i]}', bounding_box[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # 番号をふる
    
    # 画像の保存
    results_path = './results/OCR' 
    cv2.imwrite(f'{results_path}/OCR_{file_name}.png', image_OCR)
    cv2.imwrite(f'img_OCR_att.png', image_OCR) # 確認用
    


if __name__ == "__main__":
    main()