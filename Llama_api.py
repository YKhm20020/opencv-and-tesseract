import os
from typing import List, Tuple
from PIL import Image
import numpy as np
import replicate
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


def main():
    
    input_path = './sample/sample4.jpg'
    # ファイルが存在しない場合、プログラムを終了する
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return

    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    txts, bounding_boxes = extract_text(input_path, filename)
    
    atts = []
    for i in tqdm(range(len(txts))):
        
        # 〇〇というラベルは、日付、整数、文字列、単一選択、複数選択のうち、どれにあたる？
        output = replicate.run(
            "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
            input={"prompt": f'Which of the following is the label of "{txts[i]}" in Japanese ?',
                "system_prompt": 'Answer only in date, number, string, single selection or multiple selection'}
        )
        

        # The replicate/llama-2-70b-chat model can stream output as it's running.
        # The predict method returns an iterator, and you can iterate over that output.
        output_list = list(output)
        output_str = "".join(output_list)
        print(output_str)

        labels = ['date', 'number', 'string', 'single selection', 'multiple selection']
        
        atts.append(output_str)
        
    for i in range(len(atts)):
        print(f'att[{i}]: {atts[i]} ({txts[i]})')
            
        

    for i in range (len(labels)):
        if labels[i] in output_str:
            label = labels[i]
            print(f'The label of input is {label}')
            break
        

if __name__ == "__main__":
    main()
        