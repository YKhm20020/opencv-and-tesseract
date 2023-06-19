import os
import pyocr
import pyocr.builders
import pyocr.tesseract
import cv2
from PIL import Image
import sys

# インストール済みのTesseractへパスを通す
TESSERACT_PATH = "/workspaces/OpenCV/Tesseract-OCR"
if TESSERACT_PATH not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + TESSERACT_PATH

TESSDATA_PATH = "/workspaces/OpenCV/Tesseract-OCR/tessdata"
os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH
 
# 利用可能なOCRツールを取得
tools = pyocr.get_available_tools()

for tool in tools:
    print(tool.get_name())
 
if len(tools) == 0:
    print("Do not find OCR tools")
    sys.exit(1)
 
# 利用可能なOCRツールはtesseractしか導入していないため、0番目のツールを利用
tools = pyocr.get_available_tools()
tool = tools[0]
 
# 画像から文字列を取得
img = cv2.imread("sample/sample.png")

# BGR -> グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('result1.png', img_bw)

img_bw = Image.fromarray(img_bw)

builder_list = pyocr.builders.WordBoxBuilder(tesseract_layout=6)
builder_text = pyocr.builders.TextBuilder(tesseract_layout=6) 
res = tool.image_to_string(
    img_bw,
    lang="jpn",
    builder=builder_list,
)
res_txt = tool.image_to_string(
    img_bw,
    lang="jpn",
    builder=builder_text,
)

 
# 取得した文字列を表示
res_txt = res_txt.replace(' ', '')
print(res_txt)
 
# 画像のどの部分を検出し、どう認識したかを分析
out = cv2.imread("sample/sample.png")
 
for box in res:
    print(box.content) #どの文字として認識したか
    print(box.position) #どの位置を検出したか
    cv2.rectangle(out, box.position[0], box.position[1], (0, 0, 255), 1) #検出した箇所を赤枠で囲む
 
# 検出結果の画像を表示
cv2.imwrite('img_OCR.png', out)
cv2.waitKey(0)
cv2.destroyAllWindows()