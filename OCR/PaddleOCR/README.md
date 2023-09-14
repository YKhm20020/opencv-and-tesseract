# コマンド

実行コマンド。python では bash: python: command not found のエラーが出る。  
python3 を pip install しているため。  
過去バージョンのものについては、old_version ディレクトリにまとめる予定である。

```
$ python3 OCR_paddle_color.py
```

以下は二値化処理を施した画像に対してのOCRを行うファイル。どちらがよいか（精度、処理速度などの様々な観点から）は、今後検証予定。

```
$ python3 OCR_paddle.py
```

# PaddleOCR の特徴と Tesseract-OCR との違い

PaddleOCR の特徴として、他の（オープンソース）OCR よりも比較的高速であることと、カラー画像に対しても高精度で抽出ができることである。二値化の画像処理が不要であるため、その分さらに Tesseract-OCR よりも高速である。なお、Tesseract-OCR は二値化処理を前提としており、カラー画像でも実行はできるものの、精度が悪い。  
速度については、OCR_paddle の方が Tesseract-OCR よりも 2~3 倍ほど高速。精度は要検証。体感的には OCR_paddle_color.py が最も高精度である。しかし、精度については二値化を行う OCR_paddle.py 、Tesseract-OCR 使用の OCR.py 、どれも大きく差はついていない。  
入力画像によって結果に差が開く場合があるため、要検証。  
現段階では、PaddleOCR の方が全体的に優れているという結論である。

# 修正事項

1. 適合率と再現率の測定  
修正事項ではないが、適合率と再現率を数値で具体的に測定することは必要である。特に、Tesseract-OCR との比較、二値化処理を施したもの、施していないものの3者間の比較は必須である。

2. 精度の向上のための矩形領域  
精度については、既存のプログラムを走らせるだけでも十分な精度である。間違って認識されている箇所は、文字の並びが平坦でないために、取得文字を囲む矩形の線上に文字の一部が入り、見切れて正しく認識できていないものがほとんどであった。また、矩形で文字全体を囲んでいるものの、余白が狭く、画数が多い漢字が最初ないし最後に配置されており、別の文字や記号に認識されているものもあった。  
共通して、文字取得のための矩形に問題があるため、修正することができるとさらなる精度向上が見込める。

# コマンド初回実行時のモデルダウンロード

Dockerfile からコンテナを作成後、初回の実行時に以下のような表示がされる。

```
download https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar to /root/.paddleocr/whl/det/ml/Multilingual_PP-OCRv3_det_infer/Multilingual_PP-OCRv3_det_infer.tar
100%|██████████████████████████████████████████████████████| 3.85M/3.85M [00:03<00:00, 1.28MiB/s]
download https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar to /root/.paddleocr/whl/rec/japan/japan_PP-OCRv3_rec_infer/japan_PP-OCRv3_rec_infer.tar
100%|██████████████████████████████████████████████████████| 11.4M/11.4M [00:08<00:00, 1.35MiB/s]
download https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar to /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar
100%|██████████████████████████████████████████████████████| 2.19M/2.19M [00:01<00:00, 1.11MiB/s]
```
これは、テキスト検出のモデル Text Detection Model と、テキスト認識のモデル Text Recognition Model 、テキストの向き(角度)の判別のモデル Text Angle Classification Model の3種類がインストールされるためである。作成者の環境では30秒かからずに終わったため、さほど時間はかからないはずである。  
コンテナをリビルドした際もこのインストールが割り込むが、リビルド前の実行結果とリビルド後の実行結果に違いはなかった。

# 各関数の詳細

## 入力画像の決定と読み取り
```
input_image = './sample/sample.jpg'
img = cv2.imread(input_image)
```
入力画像を input_image という変数として、読み取っている。

## 検出した文字とその位置を保存するリストを用意
```
text_list = []
box_list = []
```
最後に抽出した文字とその位置を一覧表示するためと、その際にターミナルに表示される数字と画像に埋め込む数字を一致させるために必要。空のリストを用意している。

## OCR処理
```
ocr = PaddleOCR(
    use_gpu = False, # GPUを使うのであれば True
    lang = "japan", # 言語選択。英語でのOCRなら en
    det_limit_side_len = img.shape[0], # 画像サイズが960に圧縮されることを回避するための設定
    )

result = ocr.ocr(img = img, det=True, rec=True, cls=False)
```
PaddleOCR の設定を行っている。今回は、GPU使用の是非、言語、画像サイズ圧縮の回避を設定している。

## 必要な要素の代入・格納と表示
```
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
```
文字抽出を行った回数だけ、バウンディングボックスの各頂点座標、抽出したテキストの文字を格納。格納した文字、具体的な座標の数値、自信度を表示している。また、入力画像に矩形領域と数字を割り振って表示させている。  
また、用意したリストにそれぞれ文字と位置を格納している。

## 検出結果を表示
```
cv2.imwrite('img_OCR_paddle_color.png', img)

print()
for i in range(len(text_list)):
    print("{}: {} {}".format(i, text_list[i], box_list[i]))
```
img_OCR_paddle_color.png という名前の .png ファイルで画像を保存。最後に空行と、リストに格納した文字と位置を一覧表示する。このとき、割り振られた番号は画像内に埋め込まれている数字と一致している。

# 参考文献
- [PP-OCR: A Practical Ultra Lightweight OCR System (2020)（英語論文）](https://arxiv.org/abs/2009.09941)
- [PP-OCRv2: Bag of Tricks for Ultra Lightweight OCR System (2021)（英語論文）](https://arxiv.org/abs/2109.03144)
- [日本清華同方OCR技術のご紹介](http://www.tfsoftec.co.jp/dl/%E6%B8%85%E8%8F%AF%E5%90%8C%E6%96%B9_%E6%97%A5%E6%9C%AC%E8%AA%9EOCR%E6%8A%80%E8%A1%93%E7%B4%B9%E4%BB%8B%E8%B3%87%E6%96%99.pdf)
- [PaddlePaddle/PaddleOCR（GitHubのREADME）](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/quickstart_en.md)
- [paddleocr 2.6.1.3](https://pypi.org/project/paddleocr/)
- [PaddleOCR: 最新の軽量OCRシステム](https://medium.com/axinc/paddleocr-%E6%9C%80%E6%96%B0%E3%81%AE%E8%BB%BD%E9%87%8Focr%E3%82%B7%E3%82%B9%E3%83%86%E3%83%A0-8744205f3703)
- [【PaddleOCR】Pythonで簡単に日本語OCR_その2(exe化のおまけつき)](https://qiita.com/ku_a_i/items/d4c1ce70836b8035a449)
- [[AIOCR]PaddleOCRで日本語を文字認識する](https://www.12-technology.com/2021/06/aiocrpaddleocr.html)
