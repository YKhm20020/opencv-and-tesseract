import os
import easyocr
import cv2
import numpy as np
from PIL import Image

# ディレクトリ作成、入力画像の決定と読み取り
results_path = './results'
os.makedirs(results_path, exist_ok = True)

input_image = './sample/sample.jpg'
 
# 画像から文字列を取得
img = cv2.imread(input_image)

# BGR -> グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

reader = easyocr.Reader(['ja', 'en']) # this needs to run only once to load the model into memory
result = reader.readtext(img_bw)

print(result)