import os
import sys
from typing import List, Tuple
import numpy as np
import pyocr
import pyocr.builders
import pyocr.tesseract
import cv2
from PIL import Image
from fugashi import Tagger
from prepare import create_OCR_directories, load_OCR_image
from detect_area import create_area_directories, load_area_image, process_image_rect, process_image_underline, find_rectangles, find_underlines
from export_data import export_OCR_data
from functools import reduce


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

    # BGR -> グレースケール -> ガウシアンフィルタ -> 二値化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f'{results_path}/1_thresh.png', img_bw) # 確認用
    
    # 配列を画像に変換
    img_bw = Image.fromarray(img_bw)
    
    return img_bw


def find_text_and_bounding_box(img_bw: np.ndarray, img_OCR: np.ndarray, file_name: str) -> Tuple[List[str], List[np.ndarray]]:
    """ 抽出文字とバウンディングボックスの座標を検出する関数
    
    Tesseract-OCR より、画像中の文字を抽出し、文字とそのバウンディングボックスの座標を検出する関数
    
        Args:
            img_bw (numpy.ndarray): エッジ検出後の画像
            img_OCR (numpy.ndarray): 結果出力用の画像
            file_name (str): ファイルの名前
        
        Returns:
            Tuple[List(str), List(numpy.ndarray)]: 抽出した文字、抽出文字を囲うバウンディングボックス
            
        Note:
            text_result (List[str]): 抽出した文字
            bounding_box_result (List[numpy.ndarray]): 抽出文字を囲うバウンディングボックス
            
            抽出文字とバウンディングボックスの関係性を維持してソートする箇所については、
            y 座標を 1/10 で比較することで、10px 以内のズレを同一値とみなし、左右の並び順を維持している。
    
    """
    
    # インストール済みのTesseractへパスを通す
    TESSERACT_PATH = os.path.abspath('TESSERACT-OCR')
    if TESSERACT_PATH not in os.environ['PATH'].split(os.pathsep):
        os.environ['PATH'] += os.pathsep + TESSERACT_PATH

    TESSDATA_PATH = os.path.join(TESSERACT_PATH, 'tessdata')
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH
    
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
    delete_index = []
    res_txt = res_txt.replace(' ', '') # 余分な空白を削除
    res_txt = res_txt.replace('\n\n', '\n') # 余分な改行を削除
    splitted_txt = res_txt.split('\n') # 改行で分割

    # 除外対象外と判断するホワイトリスト
    text_white_list = ['〒'] 
    #pos_white_list = ['名詞,数詞,*,*']

    # 除外対象と判断するブラックリスト
    text_black_list = ['！'] 
    pos_black_list = ['補助記号,一般,*,*', '感動詞,フィラー,*,*', '感動詞,一般,*,*']

    for i, line in enumerate(splitted_txt):
        text.append(line)
        tagger = Tagger('-Owakati')
        tagger.parse(text[i])
        result = tagger.parse(text[i])
        
        # 形態素解析によって誤検知を排除
        parts, count_symbol = 0, 0 # 形態素数と誤検知と認識した数
        for word in tagger(text[i]): 
            if word.feature.lemma in text_black_list or word.feature.lemma is None or word.pos in pos_black_list:
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
    
    # 上下誤差10ピクセルを基準にグループ化
    group_margin = 10
    grouped_indices = {}
    for i in range(len(bounding_box_result)):
        group = [] 
        for j in range(i+1, len(bounding_box_result)):
            if abs(bounding_box_result[i][0][1] - bounding_box_result[j][0][1]) <= group_margin:
                group.append(j)
        grouped_indices[i] = group

    # ソートされたインデックス
    sorted_indices = []
    processed_indices = set()
    for key, group in grouped_indices.items():
        if key not in processed_indices:
            indices = sorted([key] + group, key=lambda x: bounding_box_result[x][0][0])
            sorted_indices.extend(indices)
            processed_indices.update(indices)
    
    # それぞれをソート後の結果に更新
    text_result = [text_result[i] for i in sorted_indices]
    bounding_box_result = [bounding_box_result[i] for i in sorted_indices]
    
    # 動作結果をファイルにエクスポート
    export_OCR_data(text_result, bounding_box_result, file_name)

    return text_result, bounding_box_result



def main():
    # ディレクトリ作成
    create_OCR_directories()

    try:
        input_path = './sample/seikyuu.jpg'

        #input_path =  './sample/P/3．入出退健康管理簿.pdf'
        #input_path =  './sample/P/13-3-18 入出退健康管理簿（確認印欄あり）.pdf'
        #input_path =  './sample/P/20230826_富士瓦斯資料_設備保安点検01.pdf'
        #input_path = './sample/P/（158-000306）自動車保険契約内容変更依頼書/作成/【ベース】AA300319_1-1.jpg'
        #input_path = './sample/P/（158-000306）自動車保険契約内容変更依頼書/作成/変更_AA300319.pdf'

        #input_path = './sample/sample.png'
        #input_path = './sample/P/02稟議書_/A281新卒者採用稟議書.png'
        #input_path = './sample/P/02稟議書_/A282広告出稿稟議書.png'
        #input_path = './sample/P/02稟議書_/A321稟議書.png'
        #input_path = './sample/P/02稟議書_/A438安全衛生推進者選任稟議書.png'
        #input_path = './sample/P/02稟議書_/A481広告出稿稟議書.png'
        #input_path = './sample/P/18作業報告書_/B090入庫報告書.png'
        #input_path = './sample/P/26休暇届_/A089夏季休暇届.png'
        
        # ファイルが存在しない場合の例外処理
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"The file '{input_path}' does not exist.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit()

    file_name = os.path.splitext(os.path.basename(input_path))[0]

    # 入力画像の読み込み
    image_original, image_OCR = load_OCR_image(input_path)
    
    # 画像処理
    image_bw = process_image_OCR(image_original)

    # テキスト抽出とバウンディングボックス検出
    text, bounding_box = find_text_and_bounding_box(image_bw, image_OCR, file_name)
    
    # 画像への描画
    for i in range(len(text)):
        print(f'string[{i}] {bounding_box[i]} : {text[i]}') # 座標と文字列を出力
        cv2.rectangle(image_OCR, bounding_box[i][0], bounding_box[i][1], (0, 0, 255), 1) # 検出した箇所を赤枠で囲む
        cv2.putText(image_OCR, str(i), bounding_box[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # 番号をふる

    # 画像の保存
    results_path = './results/OCR' 
    cv2.imwrite(f'{results_path}/OCR_{file_name}.png', image_OCR)
    cv2.imwrite(f'img_OCR.png', image_OCR) # 確認用
    
    
    
if __name__ == "__main__":
    main()