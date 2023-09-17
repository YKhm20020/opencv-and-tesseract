import os
import pyocr
import pyocr.builders
import pyocr.tesseract
import cv2
import numpy as np
from PIL import Image
import sys
from fugashi import Tagger

# インストール済みのTesseractへパスを通す
TESSERACT_PATH = os.path.abspath('TESSERACT-OCR')
if TESSERACT_PATH not in os.environ['PATH'].split(os.pathsep):
    os.environ['PATH'] += os.pathsep + TESSERACT_PATH

TESSDATA_PATH = os.path.join(TESSERACT_PATH, 'tessdata')
os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH

# ディレクトリ作成、入力画像の決定と読み取り
results_path = './results'
os.makedirs(results_path, exist_ok = True)

input_image = './sample/sample2.jpg'
 
# 利用可能なOCRツールを取得
tools = pyocr.get_available_tools()

for tool in tools:
    print(tool.get_name())
 
if len(tools) == 0:
    print('Do not find OCR tools')
    sys.exit(1)

# tools[1] へ変更を検討。結果はほぼ変更がないがやや高速。入力画像によっては少し1がよいかも？　程度
tool = tools[0]
 
# 画像から文字列を取得
img = cv2.imread(input_image)

# BGR -> グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 配列を画像に変換
img_bw = Image.fromarray(img_bw)

# OCR処理
builder_list = pyocr.builders.LineBoxBuilder(tesseract_layout=11)
builder_text = pyocr.builders.TextBuilder(tesseract_layout=11) 
res = tool.image_to_string(
    img_bw,
    lang='jpn',
    builder=builder_list,
)
res_txt = tool.image_to_string(
    img_bw,
    lang='jpn',
    builder=builder_text,
)
 
# 画像のどの部分を検出し、どう認識したかを分析
out = cv2.imread(input_image)

# 取得した文字列を表示
text = []
text_result = []
delete_index = []
bounding_box_result = []
res_txt = res_txt.replace(' ', '')
res_txt = res_txt.replace('\n\n', '\n') # 余分な改行を削除
splitted_txt = res_txt.split('\n') # 改行で分割

for i, line in enumerate(splitted_txt):
    text.append(line)
    tagger = Tagger('-Owakati')
    tagger.parse(text[i])
    result = tagger.parse(text[i])
    
    
    # 形態素解析によって誤検知を排除
    parts, count_symbol = 0, 0
    for parts, word in enumerate(tagger(text[i])): 
        if word.feature.lemma != '〒' and (word.pos == '補助記号,一般,*,*' or word.pos == '感動詞,フィラー,*,*'):
            count_symbol += 1
        print(word, word.feature.lemma, word.pos, sep='\t') 
    # 一定割合以上が不要な品詞である場合、インデックスを保存。
    if count_symbol + 1 >= parts * 0.6:
        delete_index.append(i)
    else:
        text_result.append(line)
    
    
for i, box in enumerate(res):
    # 保存したインデックス番目の場合、bounding_box_result の append をスキップ
    if not i in delete_index:
        bounding_box_result.append(box.position)

for i, line in enumerate(text_result):
    print(f'chars[{i}] {bounding_box_result[i]} : {text_result[i]}') # 座標と文字列を出力
    cv2.rectangle(out, bounding_box_result[i][0], bounding_box_result[i][1], (0, 0, 255), 1) #検出した箇所を赤枠で囲む
    cv2.putText(out, str(i), bounding_box_result[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # 番号をふる

# 検出結果の画像を表示
cv2.imwrite('img_OCR.png', out)