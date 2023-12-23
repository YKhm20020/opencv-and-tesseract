from typing import List, Dict, Union
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
    formatted_data = {}
    
    for idx, underline_coords in enumerate(data):
        formatted_data[f"underline{idx}"] = {
            "left": {"x": int(underline_coords[0]), "y": int(underline_coords[1])},
            "right": {"x": int(underline_coords[2]), "y": int(underline_coords[3])}
        }
        
    with open(f'{data_path}/json/underlines_data_{file_name}.json', 'w', encoding='utf_8_sig') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)
        
    # CSV ファイルにエクスポート
    with open(f'{data_path}/csv/underlines_data_{file_name}.csv', 'w', encoding='utf_8_sig', newline='') as f:
        writer = csv.writer(f)
        # ヘッダーを書き込む
        writer.writerow(["Underline", "left_x", "left_y", "right_x", "right_y"])
        
        # データを書き込む
        for idx, underline_coords in enumerate(data):
            writer.writerow([f"underline{idx}"] + [int(coord) for coord in underline_coords])

       
def export_OCR_data(txt: List[str], b_box: List[np.ndarray], file_name: str) -> None:
    """ 実行結果をファイルにエクスポートする関数
    
    検出したテキストとそのバウンディングボックスの座標を、json, csv ファイルとしてエクスポートする関数
    
        Args:
            txt (List[str]): 抽出した文字
            b_box (List[numpy.ndarray]): 抽出文字を囲うバウンディングボックスの座標
            file_name (str): 元画像のファイル名
    
    """
    
    def format_bounding_box(bounding_box: np.ndarray) -> Dict[str, Dict[str, Union[int, int]]]:
        """バウンディングボックスを指定の形式に整形する関数
        
        Args:
            bounding_box (numpy.ndarray): バウンディングボックスの座標
        
        Returns:
            Dict[str, Dict[str, Union[int, int]]]: 整形されたバウンディングボックス
        """
        return {
            "top_left": {"x": int(bounding_box[0][0]), "y": int(bounding_box[0][1])},
            "bottom_right": {"x": int(bounding_box[1][0]), "y": int(bounding_box[1][1])}
        }

    
    data_path = './data/OCR'
    
    # 辞書形式に整形
    ocr_data = [{"text": t, "bounding_box": format_bounding_box(bb)} for t, bb in zip(txt, b_box)]
  
    # JSON ファイルにエクスポート
    with open(f'{data_path}/json/ocr_data_{file_name}.json', 'w', encoding='utf_8_sig') as f:
        json.dump(ocr_data, f, ensure_ascii=False, indent=4)
    
    
    # CSV ファイルにエクスポート
    with open(f'{data_path}/csv/ocr_data_{file_name}.csv', 'w', encoding='utf_8_sig', newline='') as f:
        writer = csv.writer(f)
        # ヘッダーを書き込む
        writer.writerow(["Text", "Top Left X", "Top Left Y", "Bottom Right X", "Bottom Right Y"])
        
        # データを書き込む
        for t, bb in zip(txt, b_box):
            top_left = bb[0]
            bottom_right = bb[1]
            writer.writerow([t, top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
            

def export_label_data(r_labels: str, rects: np.ndarray, u_labels: str, underlines: List[np.ndarray], file_name: str) -> None:
    """ 実行結果をファイルにエクスポートする関数
    
    付与したラベルとその領域の座標を、json, csv ファイルとしてエクスポートする関数
    
        Args:
            r_labels (str): 矩形領域に付与したラベル
            rects (numpy.ndarray): 矩形領域の頂点座標を格納したリスト
            u_labels (str): 下線部領域に付与したラベル
            underlines (List[numpyp.ndarray]): 下線の両端点の座標を格納したリスト
            file_name (str): 元画像のファイル名
    
    """
    
    data_path = './data/labels'
    
    # 辞書形式に整形
    rect_label_data = [{"rect_label": r_label, "rect": r} for r_label, r in zip(r_labels, rects)]
    underline_label_data = [{"rect_label": u_label, "underline": u} for u_label, u in zip(u_labels, underlines)]
    
    # JSON ファイルにエクスポート
    with open(f'{data_path}/json/labels_data_{file_name}.json', 'w', encoding='utf_8_sig') as f:
        json.dump({"rects_data": rect_label_data, "underlines_data": underline_label_data}, f, indent=4, ensure_ascii=False)