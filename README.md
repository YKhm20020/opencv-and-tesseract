# はじめに

本システムは、スマートフォンで撮影した紙媒体の書類フォームをデジタル化する、というコンセプトのもとで作成している卒業研究のものである。  
具体的には、入力画像を二値化した後に矩形領域を取得するという内容である。

# 実行環境
- OS: Window8, Windows10

# export_array について

export_array という名前のファイルは、矩形領域取得に関するシステムである。詳細は、export_array ディレクトリの README.md を参照。  
以下では、簡単に実行方法と出力画像の説明を行う。

```
python3 export_array.py
```

Dockerfile によってコンテナ作成後、上のコマンドで export_array.py を実行する。実行後、img.png という.png ファイルが生成されると同時に、ターミナルに番号と 4 つの数字が出力される。  
出力画像にはランダムの色で取得した矩形領域の枠が色付けられ、同じ色で番号が振られる。この番号は、ターミナルに出力された番号と一致する。  
4 つの数字は、左上を原点として、対象の矩形領域について[x座標 y座標]というように、各頂点の座標を示している。
~~左下、左上、右上、右下の順が多いが、一部は左上、右上、右下、左下の順で出力されているときもある。~~  
最新バージョンにて、左上、左下、右下、右上と、左上から反時計回りに座標が並ぶよう修正済み。以下はターミナルにおける出力の例である。

```
rect(0):
 [[2231  205]
 [2606  199]
 [2614  705]
 [2239  711]]
rect(1):
 [[ 284  366]
 [2110  351]
 [2111  454]
 [ 285  469]]
 .
 .
 .
```

# OCR について

OCR という名前のファイルは、光学文字認識(Optical Character Recognition)に関するシステムである。詳細は、各 OCR ディレクトリの README.md を参照。  
以下では、簡単に実行方法と出力画像の説明を行う。OCRについては、Tesseract-OCR を用いたものと、PaddleOCR を用いたものの2つに分けて説明する。  

## Tesseract-OCR について
最初に、Tesseract-OCR を用いた OCR.py についての説明を行う。詳細は、OCR/Tesseract ディレクトリの README.md を参照。

```
python3 OCR.py
```

Dockerfile によってコンテナ作成後、上のコマンドで OCR.py を実行する。実行後、img_OCR.png という.png ファイルが生成されると同時に、ターミナルに文字と 4 つの数字が出力される。  
出力画像に認識した文字枠が赤色で色付けられる。4 つの数字は、対象の赤枠の左上の点のx, y座標、右下の点のx, y座標の順にそれぞれの角が何ピクセルにあたるかを示している。わかりやすいよう、最後に抽出した文字のみを再表示している。以下は、ターミナルにおける出力の例である。  
なお、最初の OCR ツールの表示は削除予定である。

```
Tesseract (sh)
Tesseract (C-API)
履 歴
((279, 225), (504, 317))
圭
((557, 222), (646, 279))
過

.
.
.

横24一30mm

2.本人単身胸から上

氏名

3.裏面のりづけ

ふりがな
```

## PaddleOCR について
次に、PaddleOCR を用いた OCR_paddle.py, OCR_paddle_color.py についての説明を行う。詳細は、OCR/PaddleOCR ディレクトリの README.md を参照。  
両者ファイルとも PaddleOCR を用いて文字とその位置を抽出しているが、前者は入力画像に対して二値化処理を行った上で抽出を行うファイルであり、後者は入力画像そのままに対して抽出を行うファイルである。  
実行のコマンドは、それぞれ以下の通りである。

```
python3 OCR_paddle.py
```

または

```
python3 OCR_paddle_color.py
```

Dockerfile によってコンテナ作成後、上のコマンドで OCR_paddle.py, OCR_paddle_color を実行する。実行後、img_OCR_paddle.png または img_OCR_paddle_color という.png ファイルが生成されると同時に、ターミナルに文字と 4 つの数字に加え、0~1 の浮動小数点数が出力される。  
出力画像に認識した文字枠が赤色で色付けられる。4 つの数字は、対象の赤枠の左上の点のx, y座標、右下の点のx, y座標の順にそれぞれの角が何ピクセルにあたるかを示している。確認用として、最後に抽出した文字とその位置を再表示している。このときの左の数字は、出力画像における矩形領域付近にある数字と一致している。浮動小数点数は、抽出した文字が合っているかどうかを示したものの推論の値である。この値が高ければ高いほど、精度が高くなる傾向にある。  
以下は、ターミナルにおける出力の例である。なお、二者ファイル間で出力そのものは変化するが、出力の形式は統一している。  
また、実行時にファイルのインストールが入るが、これは実行に必要なモデルをインストールしているためである。詳細は OCR/Paddle ディレクトリの README.md を参照。
なお、以下で出力された文字列には、普通に読むと誤字脱字等とみなされるものがあるが、これは私のタイプミスによるものではなく、文字抽出の誤りであり、抽出した内容をそのままペーストしている。  

```
履歴書
(277, 224, 651, 318)
0.9105887413024902
年月日
(1432, 256, 1764, 307)
0.9936679005622864
現在
(1819, 254, 1914, 304)
0.9999313354492188
写真を貼る位置
(2295, 276, 2542, 309)
0.9993799924850464

・
・
・

検出した文字とその位置は以下の通りです。
0: 履歴書 (277, 224, 651, 318)
1: 年月日 (1432, 256, 1764, 307)
2: 現在 (1819, 254, 1914, 304)
・
・
・
32: [性別 (311, 3797, 437, 3839)
33: 欄：記載は任意です (464, 3796, 879, 3839)
34: 未記載とすることも可能です。 (912, 3799, 1517, 3826)
```


## export_array_ex.py, OCR_ex.py について
export_array_ex.py という名前のファイルは、 export_array と同じく矩形領域を検出する機能に加え、新たにその矩形領域ごとに画像を切り取る機能を追加したものである。  
現段階ではまだ使うことはないが、調査したところ、画像を文字ごとに切り取ると OCR の精度がより向上するようなので、先に実装を進めた。  
しかし、Tesseract-OCR にこの処理を適応させた OCR_ex.py を作成・実行したものの、精度は明確に低下した。　
PaddleOCR については、もともとの精度が高く、矩形領域内のみ精度を向上させたとして、入力画像によっては矩形領域の数が少ないなど、恩恵が少ないことが考えられるため、あまり役に立たないという見込みである。
