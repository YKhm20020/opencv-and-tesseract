import sys
import cv2
import pyocr
import pyocr.builders
import numpy as np
from PIL import Image, ImageEnhance
import os

TESSERACT_PATH = "C:\\Program Files\\Tesseract-OCR"
TESSDATA_PATH = "C:\\Program Files\\Tesseract-OCR\\tessdata" #tessdataのpath

os.environ["PATH"] += os.pathsep + TESSERACT_PATH
os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH

image = "./sample/sample.png"
name = "test"

#OCRエンジン取得
tools = pyocr.get_available_tools()
tool = tools[0]

#OCRの設定 ※tesseract_layout=6が精度には重要。デフォルトは3
builder = pyocr.builders.TextBuilder(tesseract_layout=6)

#解析画像読み込み(雨ニモマケズ)
img = Image.open(image) #他の拡張子でもOK

#適当に画像処理(何もしないと結構制度悪いです・・)
img_g = img.convert('L') #Gray変換
enhancer= ImageEnhance.Contrast(img_g) #コントラストを上げる
img_con = enhancer.enhance(2.0) #コントラストを上げる

#画像からOCRで日本語を読んで、文字列として取り出す
txt_pyocr = tool.image_to_string(img_con , lang="jpn", builder=builder)

#半角スペースを消す ※読みやすくするため
txt_pyocr = txt_pyocr.replace(' ', '')

print(txt_pyocr)

'''

# original
img = cv2.imread(image)
cv2.imwrite(f"1_{name}_original.png", img)

# gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f"2_{name}_gray.png", img)

# threshold
th = 140
img = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite(f"3_{name}_threshold_{th}.png", img)

# bitwise
img = cv2.bitwise_not(img)
cv2.imwrite(f"4_{name}_bitwise.png", img)

cv2.imwrite("target.png", img)

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
tool = tools[0]
res = tool.image_to_string(Image.open("target.png"), lang="jpn")

print(res)
'''
