import os
import sys
from typing import List, Tuple
import numpy as np
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from prepare import create_label_directories
from detect_area import create_area_directories, load_area_image, process_image_rect, process_image_underline, find_rectangles, find_underlines
from OCR import create_OCR_directories, load_OCR_image, process_image_OCR, find_text_and_bounding_box
from predict_text_att import link_attribute_to_text
from export_data import export_label_data

def area_detection(input_img: np.ndarray, filename: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """ 領域検出機能を動作させる関数
    
    ディレクトリ作成から矩形領域と下線部領域の座標を検出するまでの処理をまとめた関数
    
        Args:
            input_img (numpy.ndarray): 入力画像
            filename (str): ファイルの名前
        
        Returns:
            Tuple[numpy.ndarray, List(numpy.ndarray)]: 矩形領域の各頂点の x, y 座標、下線部の両端点の x, y 座標
    
    """
    
    # ディレクトリ作成
    create_area_directories()
    
    # 入力画像の読み込み
    img_area_original, img_rects, img_underline = load_area_image(input_img)

    # 画像処理
    img_area_bw, retval = process_image_rect(img_area_original)
    img_area_bw_inv, retval = process_image_underline(img_area_original)
    
    # 領域取得
    rect_coords = find_rectangles(img_area_bw, img_rects, filename)
    underline_coords = find_underlines(img_area_bw_inv, img_underline, rect_coords, retval, filename)
    
    return rect_coords, underline_coords


def text_extraction(input_img: np.ndarray, filename: str) -> Tuple[List[str], List[np.ndarray]]:
    """ 文字抽出機能を動作させる関数
    
    ディレクトリ作成から文字の抽出とバウンディングボックスの座標を検出するまでの処理をまとめた関数
    
        Args:
            input_img (numpy.ndarray): 入力画像
            filename (str): ファイルの名前
        
        Returns:
            Tuple[List[str], List(numpy.ndarray)]: 矩形領域の各頂点の x, y 座標、下線部の両端点の x, y 座標
    
    """

    # ディレクトリ作成
    create_OCR_directories()
    
    # 入力画像の読み込み
    img_OCR_original, img_OCR = load_OCR_image(input_img)
    
    # 画像処理
    img_OCR_bw = process_image_OCR(img_OCR_original)
    
    # テキスト抽出とバウンディングボックス検出
    txts, b_boxes = find_text_and_bounding_box(img_OCR_bw, img_OCR, filename)
    
    return txts, b_boxes


def label_prediction(rects: np.ndarray, underlines: List[np.ndarray], txts: str, b_boxes: List[np.ndarray], file_name: str) -> Tuple[List[str], List[str]]:
    """ 領域のラベルを決定する関数
    
    推測した文字の属性から、領域のラベルを決定する関数。
    
        Args:
            rects (numpy.ndarray): 矩形領域の各頂点の x, y 座標
            underlines (List[numpy.ndarray]): 下線の両端点の座標
            txts (str): 抽出した文字
            b_boxes (List[numpy.ndarray]): 抽出文字を囲うバウンディングボックス
            file_name (str): 元画像のファイル名
        
        Returns:
            Tuple[List[str], List[str]]: 矩形領域のラベル、下線部領域のラベル
            
        Note:
            rect_labels: 矩形領域のラベル
            underline_labels: 下線部領域のラベル
    
    """

    # モデルの読み込み
    tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b-chat-gptq")
    model = AutoGPTQForCausalLM.from_quantized("rinna/youri-7b-chat-gptq", use_safetensors=True) 
    
    text_atts = link_attribute_to_text(tokenizer, model, txts)
    
    # バウンディングボックスの中心点の座標
    b_box_centers = [((b_box[0][0] + b_box[1][0]) / 2, (b_box[0][1] + b_box[1][1]) / 2) for b_box in b_boxes]
    
    # 矩形領域のラベル付け
    if rects is None:
        rect_labels = None
        print('There are no rects and the labels')
    else:
        rect_labels = ['string' for _ in rects]
        for i in range(len(b_box_centers)):
            for j in range(len(rects)):
                if b_box_centers[i][0] < rects[j][2][0] and b_box_centers[i][1] < rects[j][2][1]:
                    rect_labels[j] = text_atts[i]
    
    # 下線部領域のラベル付け
    if underlines is None:
        underline_labels = None
        print('There are no underlines and the labels')
    else:
        underline_labels = ['string' for _ in underlines]
        for i in range(len(b_box_centers)):
            for j in range(len(underlines)):
                if b_box_centers[i][0] < underlines[j][0] and b_box_centers[i][1] < underlines[j][1]:
                    underline_labels[j] = text_atts[i]
                    
    export_label_data(rect_labels, rects, underline_labels, underlines, file_name)
    
    return rect_labels, underline_labels

    
def main():
    # ディレクトリ作成
    create_label_directories()
    
    try:
        input_path = './sample/sample4.jpg'
        #input_path = './sample/sample.png'
        
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
    
    rect_coordinates, underline_coordinates = area_detection(input_path, file_name)
    texts, bounding_box_coordinates = text_extraction(input_path, file_name)
    
    print('\nstart label prediction')
    rect_area_labels, underline_area_labels = label_prediction(rect_coordinates, underline_coordinates, texts, bounding_box_coordinates, file_name)
    
    print()
    
    if rect_coordinates is None:
        print('No rect labels because there are no rects')
    else:
        for i, label in enumerate(rect_area_labels):
            print(f'rect_label[{i}]: {label}')
        
    print()
    
    if underline_area_labels is None:
        print('No underline labels because there are no underlines')
    else:
        for i, label in enumerate(underline_area_labels):
            print(f'underline_label[{i}]: {label}')
    


if __name__ == "__main__":
    main()
    
