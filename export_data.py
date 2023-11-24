from typing import List
import numpy as np
import json
import csv

def export_area_data(data: List[np.ndarray], file_path: str, file_format: str) -> None:
    """ 実行結果をファイルにエクスポートする関数
    
    検出したテキストとそのバウンディングボックスの座標を、txt, json, csv ファイルとしてエクスポートする関数
    
        Args:
            data (List[np.ndarray]): 領域の座標を格納したリスト
            file_path (str): エクスポートするファイルのパス
            file_format (str): エクスポートするファイルの拡張子

    """
    if file_format == 'txt':
        with open(file_path, 'w') as f:
            json.dump(data, f)
    elif file_format == 'json':
        with open(file_path, 'w') as f:
            json.dump(data, f)
    elif file_format == 'csv':
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

       
def export_OCR_data(file_path: str, text: List[str], bounding_box: List[np.ndarray]) -> None:
    """ 実行結果をファイルにエクスポートする関数
    
    検出したテキストとそのバウンディングボックスの座標を、txt, json, csv ファイルとしてエクスポートする関数
    
        Args:
            file_path (str): エクスポートするファイルのパス
            text (List[str]): 抽出した文字
            bounding_box (List[numpy.ndarray]): 抽出文字を囲うバウンディングボックスの座標
    
    """
    # .txt, .json, .csv ファイルで抽出した文字をエクスポート
    with open(f'{file_path}/txt/string_text_data.txt', 'w', encoding='utf_8_sig') as f:
        json.dump(text, f)

    with open(f'{file_path}/json/string_text_data.json', 'w', encoding='utf_8_sig') as f:
        json.dump(text, f)
        
    with open(f'{file_path}/csv/string_text_data.csv', 'w', encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerow(text)
        
    # .txt, .json, .csv ファイルで文字位置を示すバウンディングボックスの座標をエクスポート
    with open(f'{file_path}/txt/string_bounding_box_data.txt', 'w') as f:
        json.dump(bounding_box, f)

    with open(f'{file_path}/json/string_bounding_box_data.json', 'w') as f:
        json.dump(bounding_box, f)

    with open(f'{file_path}/csv/string_bounding_box_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(bounding_box)