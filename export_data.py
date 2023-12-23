from typing import List
import numpy as np
import json
import csv

def export_rects_data(data: np.ndarray, file_name: str) -> None:
    """ 実行結果をファイルにエクスポートする関数
    
    矩形領域の各頂点の座標を、json, csv ファイルとしてエクスポートする関数
    
        Args:
            data (numpy.ndarray): 矩形領域の頂点座標を格納したリスト
            file_name (str): 元画像のファイル名

    """
    
    data_path = './data/rects'
    
    # JSON ファイルにエクスポート
    labels = ["top_left", "bottom_left", "bottom_right", "top_right"]
    formatted_data = {}
    
    for idx, rect_coords in enumerate(data):
        formatted_data[f"rect{idx}"] = {
            f"{labels[point_idx]}": {"x": int(x), "y": int(y)} for point_idx, (x, y) in enumerate(rect_coords)
        }
            
    with open(f'{data_path}/json/rects_data_{file_name}.json', 'w', encoding='utf_8_sig') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)
        
    # CSV ファイルにエクスポート
    with open(f'{data_path}/csv/rects_data_{file_name}.csv', 'w', encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        # ヘッダーを書き込む
        writer.writerow(["Rectangle", "Top_Left_x", "Top_Left_y", "Bottom_Left_x", "Bottom_Left_y", "Bottom_Right_x", "Bottom_Right_y", "Top_Right_x", "Top_Right_y"])
        
        # データを書き込む
        for idx, rect_coords in enumerate(data):
            writer.writerow([f"rect{idx}"] + [int(coord) for point in rect_coords for coord in point])
        
        writer.writerow(data)


def export_underlines_data(data: List[np.ndarray], file_name: str) -> None:
    """ 実行結果をファイルにエクスポートする関数
    
    検出したテキストとそのバウンディングボックスの座標を、json, csv ファイルとしてエクスポートする関数
    
        Args:
            data (List[np.ndarray]): 下線の両端点の座標を格納したリスト
            file_name (str): 元画像のファイル名

    """
    
    data_path = './data/underlines'
    
    # JSON ファイルにエクスポート
    labels = ["left_x", "left_y", "right_x", "right_y"]
    formatted_data = {}
    
    print(data)
    
    for idx, underline_coords in enumerate(data):
        formatted_data[f"underline{idx}"] = {
            labels[i]: int(underline_coords[i]) for i in range(len(labels))
        }
        
    with open(f'{data_path}/json/underlines_data_{file_name}.json', 'w', encoding='utf_8_sig') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)
        
    # CSV ファイルにエクスポート
    with open(f'{data_path}/csv/underlines_data_{file_name}.csv', 'w', encoding='utf_8_sig', newline='') as f:
        writer = csv.writer(f)
        # ヘッダーを書き込む
        writer.writerow(["Underline"] + labels)
        # データを書き込む
        for idx, underline_coords in enumerate(data):
            writer.writerow([f"underline{idx}"] + [int(coord) for coord in underline_coords])        
        writer.writerow(data)

       
def export_OCR_data(txt: List[str], b_box: List[np.ndarray], file_name: str) -> None:
    """ 実行結果をファイルにエクスポートする関数
    
    検出したテキストとそのバウンディングボックスの座標を、json, csv ファイルとしてエクスポートする関数
    
        Args:
            txt (List[str]): 抽出した文字
            b_box (List[numpy.ndarray]): 抽出文字を囲うバウンディングボックスの座標
            file_name (str): 元画像のファイル名
    
    """
    
    data_path = './data/OCR'
    
    # 辞書形式に整形
    ocr_data = [{"text": t, "bounding_box": bb} for t, bb in zip(txt, b_box)]
    
    # JSON ファイルにエクスポート
    with open(f'{data_path}/json/ocr_data_{file_name}.json', 'w', encoding='utf_8_sig') as f:
        json.dump(ocr_data, f, ensure_ascii=False, indent=4)
    
    
    # CSV ファイルにエクスポート
    with open(f'{data_path}/csv/ocr_data_{file_name}.csv', 'w', encoding='utf_8_sig', newline='') as f:
        writer = csv.writer(f)
        # ヘッダーを書き込む
        writer.writerow(["Text", "Bounding Box"])
        
        # データを書き込む
        for t, bb in zip(txt, b_box):
            writer.writerow([t, bb])