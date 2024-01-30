import os
import sys
import time
from typing import List, Tuple
import numpy as np
import cv2
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
    
    return img_rects, img_underline, rect_coords, underline_coords


def text_extraction(input_img: np.ndarray, filename: str, rect_coords: np.ndarray, underline_coords) -> Tuple[List[str], List[np.ndarray]]:
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
    
    # 各ピクセルの RGB 値を取得
    r, g, b = img_OCR_original[:,:,0], img_OCR_original[:,:,1], img_OCR_original[:,:,2]

    # RGB値のヒストグラムを計算
    r_hist, r_bins = np.histogram(r, 256)
    g_hist, g_bins = np.histogram(g, 256) 
    b_hist, b_bins = np.histogram(b, 256)

    # ヒストグラムで最も多い値を背景色として取得
    background_r = r_bins[np.argmax(r_hist)]
    background_g = g_bins[np.argmax(g_hist)] 
    background_b = b_bins[np.argmax(b_hist)]

    bg_color = (int(background_r), int(background_g), int(background_b))
    
    # 矩形領域と下線部領域を白に塗りつぶす
    for i, rect in enumerate(rect_coords):
        #cv2.drawContours(img_OCR_original, rect_coords, i, (255, 255, 255), 10)
        cv2.drawContours(img_OCR_original, rect_coords, i, bg_color, 15)
        
    for i in range(len(underline_coords)):
        x1, y1, x2, y2 = underline_coords[i]
        #cv2.line(img_OCR_original, (x1, y1), (x2, y2), (255, 255, 255), 10)
        cv2.line(img_OCR_original, (x1, y1), (x2, y2), bg_color, 15)
        
    results_path = './results/labels' 
    cv2.imwrite(f'{results_path}/1_before_OCR.png', img_OCR_original) # 確認用
    
    # 画像処理
    img_OCR_bw = process_image_OCR(img_OCR_original)
    
    # テキスト抽出とバウンディングボックス検出
    txts, b_boxes = find_text_and_bounding_box(img_OCR_bw, img_OCR, filename)
    
    # 画像への描画
    for i in range(len(txts)):
        print(f'string[{i}] {b_boxes[i]} : {txts[i]}') # 座標と文字列を出力
        cv2.rectangle(img_OCR_original, b_boxes[i][0], b_boxes[i][1], (0, 0, 255), 1) # 検出した箇所を赤枠で囲む
        cv2.putText(img_OCR_original, str(i), b_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # 番号をふる
        
    cv2.imwrite(f'{results_path}/2_after_OCR.png', img_OCR_original) # 確認用
    
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
                if b_box_centers[i][0] < underlines[j][2] and b_box_centers[i][1] < underlines[j][3]:
                    underline_labels[j] = text_atts[i]
                    
    export_label_data(rect_labels, rects, underline_labels, underlines, file_name)
    
    return rect_labels, underline_labels

    
def main():
    # 実行時間の計測開始
    time_start = time.perf_counter()
    
    # ディレクトリ作成
    create_label_directories()
    
    try:
        #input_path = './sample/seikyuu.jpg'
        #input_path = './sample/nouhin.jpg'
        input_path = './sample/seikyuu_camera.jpg'
        #input_path = './sample/kensyuusyo.jpg'
        #input_path = './sample/sample4.jpg'
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
    
    image_rects, image_underlines, rect_coordinates, underline_coordinates = area_detection(input_path, file_name)
    texts, bounding_box_coordinates = text_extraction(input_path, file_name, rect_coordinates, underline_coordinates)
    
    print('\nstart label prediction')
    rect_area_labels, underline_area_labels = label_prediction(rect_coordinates, underline_coordinates, texts, bounding_box_coordinates, file_name)
    
    print()
    
    if rect_coordinates is None:
        print('No rect labels because there are no rects')
    else:
        for i, label in enumerate(rect_area_labels):
            color = np.random.randint(0, 255, 3).tolist()
            cv2.drawContours(image_rects, rect_coordinates, i, color, 2)
            cv2.putText(image_rects, f'{str(i)}: {rect_area_labels[i]}', tuple(rect_coordinates[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            print(f'rect_label[{i}]: {label}')
    
        results_path = './results/rects'
        cv2.imwrite(f'{results_path}/rects_{file_name}.png', image_rects) # 結果を描画した画像の保存
        cv2.imwrite('img_rect_labels.png', image_rects) # 一時確認用
        
    print()
    
    if underline_coordinates is None:
        print('No underline labels because there are no underlines')
    else:
        for i, label in enumerate(underline_area_labels):
            x1, y1, x2, y2 = underline_coordinates[i]
            cv2.line(image_underlines, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_underlines, f'{str(i)}: {underline_area_labels[i]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            print(f'underline_label[{i}]: {label}')
            
        results_path = './results/underlines'
        cv2.imwrite(f'{results_path}/underline_{file_name}.png', image_underlines)
        cv2.imwrite('img_underline_labels.png', image_underlines) # 確認用

    # 実行時間の計測終了
    time_end = time.perf_counter()

    # 実行時間の計算と表示
    execution_time = time_end - time_start
    print(f"\nexecution time: {execution_time}sec")

    


if __name__ == "__main__":
    main()
    
