import os
import sys
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from detect_area import create_area_directories, load_area_image, process_image_area, find_rectangles, find_underlines
from OCR import create_OCR_directories, load_OCR_image, process_image_OCR, find_text_and_bounding_box
from predict_text_att import link_attribute_to_text

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
    img_area_bw, img_area_edges, retval = process_image_area(img_area_original)
    
    # 領域取得
    rect_coords = find_rectangles(img_area_bw, img_rects, filename)
    underline_coords = find_underlines(img_area_edges, img_underline, rect_coords, retval, filename)
    
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


def label_prediction(rects, txts, b_boxes):
    
    # # モデルの読み込み
    # tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b-chat-gptq")
    # model = AutoGPTQForCausalLM.from_quantized("rinna/youri-7b-chat-gptq", use_safetensors=True)
    
    rect_labels = ['text' for _ in rects]
    underline_labels = ['text' for _ in b_boxes]
    
    # text_atts = link_attribute_to_text(tokenizer, model, txts)
    
    # b_box_centers = []
    
    # for i, rect in enumerate(rects):
        
    #     if rect[0][0] b_boxes
    
    return rect_labels, underline_labels

    
def main():
    try:
        #input_path = './sample/sample4.jpg'
        input_path = './sample/sample.png'
        
        #input_path =  './sample/P/3．入出退健康管理簿.pdf'
        #input_path =  './sample/P/13-3-18 入出退健康管理簿（確認印欄あり）.pdf'
        #input_path =  './sample/P/20230826_富士瓦斯資料_設備保安点検01.pdf'
        
        # ファイルが存在しない場合の例外処理
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"The file '{input_path}' does not exist.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit()

    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    rect_coordinates, underline_coordinates = area_detection(input_path, filename)
    texts, bounding_box_coordinates = text_extraction(input_path, filename)
    
    rect_area_labels, underline_area_labels = label_prediction(rect_coordinates, texts, bounding_box_coordinates)
    
    print()
    
    if rect_coordinates is not None:
        for i, label in enumerate(rect_area_labels):
            print(f'rect_label[{i}]: {label}')
        
    print()
    
    if underline_area_labels is not None:
        for i, label in enumerate(underline_area_labels):
            print(f'underline_label[{i}]: {label}')
    


if __name__ == "__main__":
    main()
    
