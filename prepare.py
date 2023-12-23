import os
import sys
from typing import Tuple
import cv2
from pdf2image import convert_from_path
import numpy as np

def create_area_directories() -> None:
    """ 領域取得に関するディレクトリを作成する関数
    
    領域取得の実行結果としてエクスポートするディレクトリや、処理後の画像を格納するディレクトリを作成する

    """
    os.makedirs('./results/rects', exist_ok = True)
    os.makedirs('./results/underlines', exist_ok=True)
    os.makedirs('./data/rects/json', exist_ok=True)
    os.makedirs('./data/rects/csv', exist_ok=True)
    os.makedirs('./data/underlines/json', exist_ok=True)
    os.makedirs('./data/underlines/csv', exist_ok=True)
    

def create_OCR_directories() -> None:
    """ 文字抽出に関するディレクトリを作成する関数
    
    OCRの実行結果としてエクスポートするディレクトリや、処理後の画像を格納するディレクトリを作成する

    """
    os.makedirs('./results/OCR', exist_ok = True)
    os.makedirs('./data/OCR/txt', exist_ok=True)
    os.makedirs('./data/OCR/json', exist_ok=True)
    os.makedirs('./data/OCR/csv', exist_ok=True)
    
def create_label_directories() -> None:
    """ 文字抽出に関するディレクトリを作成する関数
    
    ラベル付与の実行結果としてエクスポートするディレクトリを作成する

    """
    os.makedirs('./data/labels/json', exist_ok=True)
    os.makedirs('./data/labels/csv', exist_ok=True)


def load_area_image(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ 領域取得機能において入力画像を読み込む関数
    
    領域取得機能において入力画像を読み込む関数（画像とPDFの入力に対応）
    
        Args:
            image_path (str): 入力画像のパス

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 入力画像、矩形領域を描画する画像、下線部領域を描画する画像
            
        Note:
            img_original (numpy.ndarray): 入力画像
            img_rects (numpy.ndarray): img_original をコピーした、矩形領域を描画する画像
            img_underlines (numpy.ndarray): img_original をコピーした、下線部領域を描画する画像

    """
    if image_path.endswith('.pdf'):
        images = convert_from_path(pdf_path=image_path, dpi=300, fmt='png')
        # リストから最初の画像を選択
        input_image = images[0]
        # PIL.Image を NumPy 配列に変換
        input_image = np.array(input_image)
        # RGB から BGR に色空間を変換
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        img_original = input_image
        img_rects = input_image.copy()
        img_underlines = input_image.copy()
    else:
        img_original = cv2.imread(image_path)
        img_rects = cv2.imread(image_path)
        img_underlines = cv2.imread(image_path)
    
    return img_original, img_rects, img_underlines


def load_OCR_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """ 文字抽出機能において入力画像を読み込む関数
    
    文字抽出機能において入力画像を読み込む関数（画像とPDFの入力に対応）
    
        Args:
            image_path (str): 入力のパス

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: 入力画像、抽出文字のバウンディングボックスを描画する画像
            
        Note:
            img_original (numpy.ndarray): 入力画像
            img_OCR (numpy.ndarray): img_original をコピーした、抽出文字のバウンディングボックスを描画する画像
            
            PDF は画像化することにより対応
        
        Todo:
            PDF の2枚目以降も処理を繰り返すよう拡張する

    """
    if image_path.endswith('.pdf'):
        images = convert_from_path(pdf_path=image_path, dpi=300, fmt='png')
        # リストから最初の画像を選択
        input_image = images[0]
        # PIL.Image を NumPy 配列に変換
        input_image = np.array(input_image)
        # RGB から BGR に色空間を変換
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        img_original = input_image
        img_OCR = input_image.copy()
    else:
        img_original = cv2.imread(image_path)
        img_OCR = cv2.imread(image_path)

    return img_original, img_OCR