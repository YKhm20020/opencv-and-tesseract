import os
import pyocr
import pyocr.builders
import pyocr.tesseract
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image
import sys
from fugashi import Tagger
# 以下、データ出力用
import json
import csv

# インストール済みのTesseractへパスを通す
TESSERACT_PATH = os.path.abspath('TESSERACT-OCR')
if TESSERACT_PATH not in os.environ['PATH'].split(os.pathsep):
    os.environ['PATH'] += os.pathsep + TESSERACT_PATH

TESSDATA_PATH = os.path.join(TESSERACT_PATH, 'tessdata')
os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH

# ディレクトリ作成、入力画像の決定と読み取り
dir = ['./results/OCR', './data/OCR/txt', './data/OCR/json', './data/OCR/csv']
results_path, data_txt_path, data_json_path, data_csv_path = dir

os.makedirs(results_path, exist_ok = True)
os.makedirs(data_txt_path, exist_ok = True)
os.makedirs(data_json_path, exist_ok = True)
os.makedirs(data_csv_path, exist_ok = True)
 
# 利用可能なOCRツールを取得
tools = pyocr.get_available_tools()

for tool in tools:
    print(tool.get_name()) # 確認用
 
if len(tools) == 0:
    print('Do not find OCR tools')
    sys.exit(1)

# tools[1] へ変更を検討。結果はほぼ変更がないがやや高速。入力画像によっては少し1がよいかも？　程度
tool = tools[0]

input_image = './sample/sample.png'

#input_image =  './sample/P/3．入出退健康管理簿.pdf'
#input_image =  './sample/P/13-3-18 入出退健康管理簿（確認印欄あり）.pdf'
#input_image =  './sample/P/20230826_富士瓦斯資料_設備保安点検01.pdf'

#input_image = './sample/sample.png'
#input_image = './sample/P/02稟議書_/A281新卒者採用稟議書.png'
#input_image = './sample/P/02稟議書_/A282広告出稿稟議書.png'
#input_image = './sample/P/02稟議書_/A321稟議書.png'
#input_image = './sample/P/02稟議書_/A438安全衛生推進者選任稟議書.png'
#input_image = './sample/P/02稟議書_/A481広告出稿稟議書.png'
#input_image = './sample/P/18作業報告書_/B090入庫報告書.png'
#input_image = './sample/P/26休暇届_/A089夏季休暇届.png'

filename = os.path.splitext(os.path.basename(input_image))[0]

if input_image.endswith('.pdf'):
    input_image = convert_from_path(pdf_path = input_image, dpi = 300, fmt = 'png')
    # リストから最初の画像を選択
    input_image = input_image[0] 
    # PIL.Image を NumPy 配列に変換
    input_image = np.array(input_image) 
    # RGB から BGR に色空間を変換
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    img, out = input_image, input_image

else:
    img = cv2.imread(input_image)
    out = cv2.imread(input_image)

# BGR -> グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_TRIANGLE + cv2.THRESH_OTSU)
cv2.imwrite(f'{results_path}/1_thresh.png', img_bw) # 確認用

# kernel = np.ones((2, 2))
# img_bw = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)

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

# 取得した文字列を表示
text = []
text_result = []
delete_index = []
bounding_box_result = []
res_txt = res_txt.replace(' ', '')
res_txt = res_txt.replace('\n\n', '\n') # 余分な改行を削除
splitted_txt = res_txt.split('\n') # 改行で分割

# 除外対象外と判断するホワイトリスト
text_white_list = ['〒'] 

# 除外対象と判断するブラックリスト
text_black_list = ['！'] 
pos_black_list = ['補助記号,一般,*,*', '感動詞,フィラー,*,*']

for i, line in enumerate(splitted_txt):
    text.append(line)
    tagger = Tagger('-Owakati')
    tagger.parse(text[i])
    result = tagger.parse(text[i])
    
    # 形態素解析によって誤検知を排除
    parts, count_symbol = 0, 0
    for parts, word in enumerate(tagger(text[i])): 
        if word.feature.lemma in text_black_list or word.pos in pos_black_list:
            count_symbol += 1
        if word.feature.lemma in text_white_list:
            count_symbol -= 1
        #print(word, word.feature.lemma, word.pos, sep='\t')  # 形態素解析確認用
    
    # 一定割合以上が不要な品詞である場合、インデックスを保存。
    if count_symbol <= parts * 0.5 or (parts == 1 and count_symbol == parts):
        text_result.append(line)
    else:
        delete_index.append(i)
    
for i, box in enumerate(res):
    # 保存したインデックス番目の場合、誤検知とみなし、抽出対象としない。
    if not i in delete_index:
        bounding_box_result.append(box.position)

for i, line in enumerate(text_result):
    print(f'string[{i}] {bounding_box_result[i]} : {text_result[i]}') # 座標と文字列を出力
    cv2.rectangle(out, bounding_box_result[i][0], bounding_box_result[i][1], (0, 0, 255), 1) # 検出した箇所を赤枠で囲む
    cv2.putText(out, str(i), bounding_box_result[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # 番号をふる

# .txt, .json, .csv ファイルで文字位置を示すバウンディングボックスの座標をエクスポート
with open(f'{data_txt_path}/string_bounding_box_data.txt', 'w') as f:
    json.dump(bounding_box_result, f)

with open(f'{data_json_path}/string_bounding_box_data.json', 'w') as f:
    json.dump(bounding_box_result, f)

with open(f'{data_csv_path}/string_bounding_box_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(bounding_box_result)
    
# .txt, .json, .csv ファイルで抽出した文字をエクスポート
with open(f'{data_txt_path}/string_text_data.txt', 'w', encoding='utf_8_sig') as f:
    json.dump(text_result, f)

with open(f'{data_json_path}/string_text_data.json', 'w', encoding='utf_8_sig') as f:
    json.dump(text_result, f)
    
with open(f'{data_csv_path}/string_text_data.csv', 'w', encoding='utf_8_sig') as f:
    writer = csv.writer(f)
    writer.writerow(text_result)

# 検出結果の画像を表示
cv2.imwrite(f'{results_path}/OCR_{filename}.png', out)