from paddleocr import PaddleOCR
import cv2

# 入力画像の決定と読み取り
input_image = './sample/sample3.png'
img = cv2.imread(input_image)

# 検出した文字とその位置を保存するリストを作成
text_list = []
box_list = []

# BGR -> グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# OCR処理
ocr = PaddleOCR(
    use_gpu = False, # GPUを使うのであれば True
    lang = "japan", # 言語選択。英語でのOCRなら en
    det_limit_side_len = img.shape[0], # 画像サイズが960に圧縮されないように必須設定
    )

# PaddleOCRでOCR ※cls(傾き設定)は矩形全体での補正である。今回は1文字1文字の補正ではないため不要
result = ocr.ocr(img = img_bw, det=True, rec=True, cls=False)

# 画像のどの部分を検出し、どう認識したかを分析
out = cv2.imread(input_image)

for detection in result[0]:
    t_left = tuple([int(i) for i in detection[0][0]]) # 左上
    # t_right = tuple([int(i) for i in detection[0][1]]) # 右上
    b_right = tuple([int(i) for i in detection[0][2]]) # 右下
    b_left = tuple([int(i) for i in detection[0][3]]) # 左下
    
    ocr_text = detection[1][0] # テキストを格納

    print(ocr_text) # どの文字として認識したか
    print(t_left + b_right) # どの位置を検出したか (左上x, 左上y, 右下x, 右下y) で出力される
    print(detection[1][1]) # 自信度（処理に必要なければ消す）
    cv2.rectangle(out, t_left, b_right, (0, 0, 255), 1) # 検出した箇所を赤枠で囲む
    cv2.putText(out, str(len(text_list)), (t_left[0], t_left[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # リストに文字と位置を追加
    text_list.append(ocr_text)
    box_list.append(t_left + b_right) # 確認用。処理では使わない。

# 検出結果の画像を表示
cv2.imwrite('img_OCR_paddle.png', out)

# 検出した文字とその位置をまとめて表示。以下確認用。
print() # 空白行
print("検出した文字とその位置は以下の通りです。")
for i in range(len(text_list)):
    print("{}: {} {}".format(i, text_list[i], box_list[i]))