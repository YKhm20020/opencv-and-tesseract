from typing import List
import numpy as np
import json
import csv

def export_rects_data(data: List[np.ndarray]) -> None:
    """ 実行結果をファイルにエクスポートする関数
    
    矩形領域の各頂点の座標を、txt, json, csv ファイルとしてエクスポートする関数
    
        Args:
            data (List[np.ndarray]): 矩形領域の頂点座標を格納したリスト

    """
    
    data_path = './data/rects'
    
    with open(f'{data_path}/txt/underlines_data.txt', 'w', encoding='utf_8_sig') as f:
        json.dump(data, f)
        
    with open(f'{data_path}/json/underlines_data.json', 'w', encoding='utf_8_sig') as f:
        json.dump(data, f)
        
    with open(f'{data_path}/csv/underlines_data.csv', 'w', encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def export_underlines_data(data: List[np.ndarray]) -> None:
    """ 実行結果をファイルにエクスポートする関数
    
    検出したテキストとそのバウンディングボックスの座標を、txt, json, csv ファイルとしてエクスポートする関数
    
        Args:
            data (List[np.ndarray]): 下線の両端点の座標を格納したリスト

    """
    
    data_path = './data/underlines'
    
    with open(f'{data_path}/txt/underlines_data.txt', 'w', encoding='utf_8_sig') as f:
        json.dump(data, f)
        
    with open(f'{data_path}/json/underlines_data.json', 'w', encoding='utf_8_sig') as f:
        json.dump(data, f)
        
    with open(f'{data_path}/csv/underlines_data.csv', 'w', encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerow(data)

       
def export_OCR_data(txt: List[str], b_box: List[np.ndarray]) -> None:
    """ 実行結果をファイルにエクスポートする関数
    
    検出したテキストとそのバウンディングボックスの座標を、txt, json, csv ファイルとしてエクスポートする関数
    
        Args:
            txt (List[str]): 抽出した文字
            b_box (List[numpy.ndarray]): 抽出文字を囲うバウンディングボックスの座標
    
    """
    
    data_path = './data/OCR'
    
    # .txt, .json, .csv ファイルで抽出した文字をエクスポート
    with open(f'{data_path}/txt/string_text_data.txt', 'w', encoding='utf_8_sig') as f:
        json.dump(txt, f)

    with open(f'{data_path}/json/string_text_data.json', 'w', encoding='utf_8_sig') as f:
        json.dump(txt, f)
        
    with open(f'{data_path}/csv/string_text_data.csv', 'w', encoding='utf_8_sig') as f:
        writer = csv.writer(f)
        writer.writerow(txt)
        
    # .txt, .json, .csv ファイルで文字位置を示すバウンディングボックスの座標をエクスポート
    with open(f'{data_path}/txt/string_bounding_box_data.txt', 'w') as f:
        json.dump(b_box, f)

    with open(f'{data_path}/json/string_bounding_box_data.json', 'w') as f:
        json.dump(b_box, f)

    with open(f'{data_path}/csv/string_bounding_box_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(b_box)