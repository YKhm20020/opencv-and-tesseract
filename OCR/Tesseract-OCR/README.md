# コマンド

実行コマンド。python では bash: python: command not found のエラーが出る。  
python3 を pip install しているため。  
過去バージョンのものについては、old_version ディレクトリにまとめる予定である。

```
$ python3 OCR.py
```

# 修正事項
1. 精度の向上  
文字認識の精度向上は必須事項。正しい文字として認識できているか、座標が正しいかの2点で精度を向上させる必要あり。

2. 歪みによる文字抽出の失敗  
ある程度の歪みであれば、デジタルのフォームの画像そのままを入力としたものと同等の精度となるようにしたい。

3. 属性付与  
export_array と合わせて、矩形領域ごとに入力されるのが数値か、テキストか、日付かなどを決められるように組み合わせたい。

# 各関数の詳細

## パスを通す

```
# インストール済みのTesseractへパスを通す
TESSERACT_PATH = '/workspaces/OpenCV/Tesseract-OCR'
if TESSERACT_PATH not in os.environ['PATH'].split(os.pathsep):
    os.environ['PATH'] += os.pathsep + TESSERACT_PATH

TESSDATA_PATH = '/workspaces/OpenCV/Tesseract-OCR/tessdata'
os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH
```

TesseractとTesseractの日本語パッケージにパスを通す。

### 修正

```
# インストール済みのTesseractへパスを通す
TESSERACT_PATH = os.path.abspath('TESSERACT-OCR')
if TESSERACT_PATH not in os.environ['PATH'].split(os.pathsep):
    os.environ['PATH'] += os.pathsep + TESSERACT_PATH

TESSDATA_PATH = os.path.join(TESSERACT_PATH, 'tessdata')
os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH
```

研究室PCと所有PCでTesseractのパスが違っていたため、合わせていない方の環境では実行できなかった。そこで、TESSERACT-OCR という名前のディレクトリを探し、絶対パスを取得。取得したパスで tessdata のパスも通すというように修正した。

## OCRツールの取得と選択

```
# 利用可能なOCRツールを取得
tools = pyocr.get_available_tools()

for tool in tools:
    print(tool.get_name())
 
if len(tools) == 0:
    print('Do not find OCR tools')
    sys.exit(1)

tool = tools[0]
```

2行目でOCRツールを取得し、4行目でそのOCRツールを表示する。7行目でツールがない場合のエラー表示、11行目でツールを選択。

## 画像処理

```
# 画像から文字列を取得
img = cv2.imread(input_image)

# BGR -> グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('result1.png', img_bw)
```

export_array.py の処理を一部利用。詳細な処理の内容は export_array ディレクトリの README.md を参照。

## OCR処理
```
# 配列を画像に変換
img_bw = Image.fromarray(img_bw)

# OCR処理
builder_list = pyocr.builders.WordBoxBuilder(tesseract_layout=6)
builder_text = pyocr.builders.TextBuilder(tesseract_layout=6) 
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
```
2行目で配列から画像に変換している。これは、OpenCVにおける画像処理の段階では配列で様々な処理を施していたが、PyOCRでは配列での処理に対応しておらず、画像に変換する必要があるためである。  
5行目と7~11行目は、単語単位でOCR処理を行い、文字の位置を検出する。  
6行目と12~16行目は、テキスト全体でOCR処理を行い、テキストそのものを抽出する。  
両者の違いは、OCR処理を行う基準である。

- TextBuilder: 文字列を認識
- WordBoxBuilder: 単語単位で文字認識 + BoundingBox
- LineBoxBuilder: 行単位で文字認識 + BoundingBox
- DigitBuilder: 数字・記号を認識
- DigitLineBoxBuilder: 数字・記号を認識 + BoundingBox

### 修正点
```
# OCR処理
builder_list = pyocr.builders.LineBoxBuilder(tesseract_layout=11)
builder_text = pyocr.builders.TextBuilder(tesseract_layout=11) 
```
pyocr.builders の builders を変更。バウンディングボックスを行単位で区切ることに。これにより、単語単位の認識だとバウンディングボックスが多数重なっていた問題をある程度解決。  
また、tesseract_layout を 11 に変更。詳細は以下を参照。

- 1 オリエンテーションとスクリプト検出(OSD)のみ。
- 2 自動ページ分割、ただしOSD、またはOCRはなし。(未実装)
- 3 完全自動ページ分割、ただしOSDなし。(未実装)
- 4 サイズの異なるテキストが1列に並んでいると仮定。
- 5 縦書きの一様なテキストブロックを想定。
- 6 一様なテキストブロックを想定。
- 7 画像を1つのテキスト行として扱う。
- 8 画像を1つの単語として扱う。
- 9 画像を円内の単一単語として扱う。
- 10 画像を1つの文字として扱う。
- 11 疎なテキスト。できるだけ多くのテキストを順不同に探す。
- 12 OSDでテキストを疎にする。
- 13 生の行。画像を1つのテキスト行として扱う。

## 認識内容を格納
```
# 画像のどの部分を検出し、どう認識したかを分析
out = cv2.imread(input_image)
 
for box in res:
    print(box.content) #どの文字として認識したか
    print(box.position) #どの位置を検出したか
    cv2.rectangle(out, box.position[0], box.position[1], (0, 0, 255), 1) #検出した箇所を赤枠で囲む
```
OpenCV利用のため、画像を配列として読み出し、その画像に対して以下の3つの操作を行う。
1. なんの文字として認識したかを表示
2. その文字がどこの位置かを表示
3. 検知した箇所を赤枠で囲む  

座標については、((左上x, 左上y), (右下x, 右下y)) という形で表示されている。  
cv2.rectangle について、書式は以下のようになっている。
```
cv2.rectangle( 入力画像, 左上座標, 右上座標, 枠の色(BGR), 線の太さ(px) )
```

## 格納した認識内容の表示
```
# 取得した文字列を表示
res_txt = res_txt.replace(' ', '')
print(res_txt)
 
# 検出結果の画像を表示
cv2.imwrite('img_OCR.png', out)
```
2行目でどの文字として認識されたかわかりやすくするため、空白を省き、3行目でそれを表示。
5行目で赤枠で囲んだ画像を img_OCR.png という名前の .png ファイルで出力する。

# 参考文献
- [PyOCRでTesseractを使う](https://blog.machine-powers.net/2018/08/04/pyocr-and-tips/#%E5%88%A9%E7%94%A8%E5%8F%AF%E8%83%BD%E3%81%AAocr-tool-%E3%81%AE%E7%A2%BA%E8%AA%8D)

- [PythonのPillowのImage.fromarrayの使い方: 配列を画像に変換する](https://yu-nix.com/archives/python-pillow-image-fromarray/)

- [文字認識を学ぶ 2日目](https://note.com/djangonotes/n/ne993a087f678)

- [(2017年12月) PythonとOpenCVをこれからやってみる -3- 文字認識(1)](https://qiita.com/R_TES_/items/0c0a7382560e1f67123b)

- [【OCR】Tesseract 読み取り精度向上のために試した3つの事とその結果](https://starsand.hateblo.jp/entry/2022/06/01/135032#%E7%94%BB%E5%83%8F%E3%81%AF%E4%BA%8C%E5%80%A4%E5%8C%96%E3%81%99%E3%82%8B)

- [【Tesseract】Pythonで簡単に日本語OCR](https://qiita.com/ku_a_i/items/93fdbd75edacb34ec610)

- [(最終回)Python + OpenCVで遊んでみる(OCR編)](https://itport.cloud/?p=8326)