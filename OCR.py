import os
import sys
from typing import List, Tuple
import pyocr
import pyocr.builders
import pyocr.tesseract
import numpy as np
import cv2
from PIL import Image
from fugashi import Tagger
from prepare import create_OCR_directories, load_OCR_image
from export_data import export_OCR_data


def process_image_OCR(input_img: np.ndarray) -> np.ndarray:
    """ 画像処理を行う関数
    
    グレースケール化、ガウシアンフィルタ適用、二値化を行う
    
        Args:
            input_img (numpy.ndarray): 入力画像

        Returns:
            numpy.ndarray: 二値化後の画像
    
    """
    img = input_img.copy()
    results_path = './results/OCR' 
    cv2.imwrite(f'{results_path}/0_original.png', img) # 確認用

    # BGR -> グレースケール -> ガウシアンフィルタ -> 二値画像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f'{results_path}/1_thresh.png', img_bw) # 確認用
    
    return img_bw


def find_text_and_bounding_box(img_bw: np.ndarray, img_OCR: np.ndarray, filename: str) -> Tuple[List[str], List[np.ndarray]]:
    """ 抽出文字とバウンディングボックスの座標を検出する関数
    
    Tesseract-OCR より、画像中の文字を抽出し、文字とそのバウンディングボックスの座標を検出する関数
    
        Args:
            img_bw (numpy.ndarray): エッジ検出後の画像
            img_OCR (numpy.ndarray): 結果出力用の画像
            filename (str): ファイルの名前
        
        Returns:
            Tuple[List(str), List(numpy.ndarray)]: 抽出した文字、抽出文字を囲うバウンディングボックス
            
        Note:
            text_result (list[str]): 抽出した文字
            bounding_box_result (list[numpy.ndarray]): 抽出文字を囲うバウンディングボックス
    
    """
    # インストール済みのTesseractへパスを通す
    TESSERACT_PATH = os.path.abspath('TESSERACT-OCR')
    if TESSERACT_PATH not in os.environ['PATH'].split(os.pathsep):
        os.environ['PATH'] += os.pathsep + TESSERACT_PATH

    TESSDATA_PATH = os.path.join(TESSERACT_PATH, 'tessdata')
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH
    results_path = './results/OCR' 
    
    # 利用可能なOCRツールを取得
    tools = pyocr.get_available_tools()

    for tool in tools:
        print(tool.get_name()) # 確認用
    
    if len(tools) == 0:
        print('Do not find OCR tools')
        sys.exit(1)

    tool = tools[0]
    
    # OCR処理
    builder_list = pyocr.builders.LineBoxBuilder(tesseract_layout=11)
    builder_text = pyocr.builders.TextBuilder(tesseract_layout=11) 
    res = tool.image_to_string(
        img_bw,
        lang='jpn',
        builder=builder_list,
    )
    res_txt = tool.image_to_string(
        img_bw,
        lang='jpn',
        builder=builder_text,
    )

    # 取得した文字列を表示
    text = []
    text_result = []
    delete_index = []
    bounding_box_result = [] # バウンディングボックスの座標を格納するリスト
    res_txt = res_txt.replace(' ', '') # 余分な空白を削除
    res_txt = res_txt.replace('\n\n', '\n') # 余分な改行を削除
    splitted_txt = res_txt.split('\n') # 改行で分割

    # 除外対象外と判断するホワイトリスト
    text_white_list = ['〒'] 

    # 除外対象と判断するブラックリスト
    text_black_list = ['！'] 
    pos_black_list = ['補助記号,一般,*,*', '感動詞,フィラー,*,*']

    for i, line in enumerate(splitted_txt):
        text.append(line)
        tagger = Tagger('-Owakati')
        tagger.parse(text[i])
        result = tagger.parse(text[i])
        
        # 形態素解析によって誤検知を排除
        parts, count_symbol = 0, 0 # 形態素数と誤検知と認識した数
        for word in tagger(text[i]): 
            if word.feature.lemma in text_black_list or word.pos in pos_black_list:
                count_symbol += 1
            if word.feature.lemma in text_white_list:
                count_symbol -= 1
            parts += 1
            # print(word, word.feature.lemma, word.pos, sep='\t')  # 形態素解析確認用

        # 一定割合以上が不要な品詞である場合、インデックスを保存。
        if count_symbol >= parts * 0.5 or (parts == 1 and count_symbol == 1):
            delete_index.append(i)

    for i, box in enumerate(res):
        box_w = box.position[1][0] - box.position[0][0] # バウンディングボックスの幅
        box_h = box.position[1][1] - box.position[0][1] # バウンディングボックスの高さ
        box_area = box_w * box_h

        # 面積が一定以上の場合、インデックスを保存
        if box_area > 300000:
            delete_index.append(i)
        
        # 面積が一定以下の場合、インデックスを保存
        if box_area < 1000:
            delete_index.append(i)

    # 保存したインデックス番目のテキストと座標を削除
    text_result = [splitted_txt[i] for i in range(len(splitted_txt)) if i not in delete_index]
    bounding_box_result = [res[i].position for i in range(len(res)) if i not in delete_index]

    for i, line in enumerate(text_result):
        print(f'string[{i}] {bounding_box_result[i]} : {text_result[i]}') # 座標と文字列を出力
        cv2.rectangle(img_OCR, bounding_box_result[i][0], bounding_box_result[i][1], (0, 0, 255), 1) # 検出した箇所を赤枠で囲む
        cv2.putText(img_OCR, str(i), bounding_box_result[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # 番号をふる

    # 検出結果の画像を表示
    cv2.imwrite(f'{results_path}/OCR_{filename}.png', img_OCR)
    cv2.imwrite(f'img_OCR.png', img_OCR) # 確認用

    return text_result, bounding_box_result



def main():
    # ディレクトリ作成、入力画像の決定と読み取り
    create_OCR_directories()

    try:
        #input_path = './sample/sample6.png'
        input_path = './sample/blur_sample3.png'

        #input_path =  './sample/P/3．入出退健康管理簿.pdf'
        #input_path =  './sample/P/13-3-18 入出退健康管理簿（確認印欄あり）.pdf'
        #input_path =  './sample/P/20230826_富士瓦斯資料_設備保安点検01.pdf'

        #input_path = './sample/sample.png'
        #input_path = './sample/P/02稟議書_/A281新卒者採用稟議書.png'
        #input_path = './sample/P/02稟議書_/A282広告出稿稟議書.png'
        #input_path = './sample/P/02稟議書_/A321稟議書.png'
        #input_path = './sample/P/02稟議書_/A438安全衛生推進者選任稟議書.png'
        #input_path = './sample/P/02稟議書_/A481広告出稿稟議書.png'
        #input_path = './sample/P/18作業報告書_/B090入庫報告書.png'
        #input_path = './sample/P/26休暇届_/A089夏季休暇届.png'
        
        # ファイルが存在しない場合の例外処理
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The file '{input_path}' does not exist.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit()
    

    filename = os.path.splitext(os.path.basename(input_path))[0]

    # 入力画像の読み込み
    image_original, image_OCR = load_OCR_image(input_path)
    
    # 画像処理と領域取得
    image_bw = process_image_OCR(image_original)

    # 配列を画像に変換
    image_bw = Image.fromarray(image_bw)
    
    # テキスト抽出とバウンディングボックス検出
    text, bounding_box = find_text_and_bounding_box(image_bw, image_OCR, filename)
    
    # 動作結果をファイルにエクスポート
    results_path = './data/OCR'
    export_OCR_data(results_path, text, bounding_box)
    
    
    
if __name__ == "__main__":
    main()