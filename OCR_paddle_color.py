from paddleocr import PaddleOCR
import cv2

# 入力画像の決定と読み取り
input_image = './sample/sample.jpg'
img = cv2.imread(input_image)

# 検出した文字とその位置を保存するリスト
text_list = []
box_list = []

# OCR処理
ocr = PaddleOCR(
    use_gpu = False, # GPUを使うのであれば True
    lang = "japan", # 言語選択。英語でのOCRなら en
    det_limit_side_len = img.shape[0], # 画像サイズが960に圧縮されることを回避するための設定
    )

# PaddleOCRでOCR ※cls(傾き設定)は矩形全体での補正である。今回は1文字1文字の補正ではないため不要
result = ocr.ocr(img = img, det=True, rec=True, cls=False)

for detection in result[0]:
    t_left = tuple([int(i) for i in detection[0][0]]) # 左上
    # t_right = tuple([int(i) for i in detection[0][1]]) # 右上
    b_right = tuple([int(i) for i in detection[0][2]]) # 右下
    b_left = tuple([int(i) for i in detection[0][3]]) # 左下
    
    ocr_text = detection[1][0] # テキストを格納

    print(ocr_text) # どの文字として認識したか
    print(t_left + b_right) # どの位置を検出したか
    print(detection[1][1]) # 自信度（処理に必要なければ消す）
    cv2.rectangle(img, t_left, b_right, (0, 0, 255), 2) # 検出した箇所を赤枠で囲む
    cv2.putText(img, str(len(text_list)), (t_left[0], t_left[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # リストに文字と位置を追加
    text_list.append(ocr_text)
    box_list.append(t_left + b_right) # 確認用。処理では使わない。

# 検出結果の画像を表示
cv2.imwrite('img_OCR_paddle_color.png', img)

# 検出した文字とその位置をまとめて表示。以下確認用。
print()
for i in range(len(text_list)):
    print("{}: {} {}".format(i, text_list[i], box_list[i]))