# コマンド
実行コマンド。pythonではbash: python: command not foundのエラーが出る。
python3をpip installしているため。
```
$ python3 export_array2.py
```

# 修正事項
1. 親領域と子領域の重複
完全な矩形領域でない場合は、大きさの異なる矩形領域の集合である。
この場合、最小の矩形領域に加えて、それらの集合である1つの大きな矩形領域が別に検出されてしまう。

# 各関数の詳細

## 画像の読み込み

```
img = cv2.imread('./sample.png')
```
imread：画像ファイルを読み込んで、多次元配列(numpy.ndarray)にする。

第一引数：画像のファイルパス
戻り値：行 x 列 x 色の三次元配列(numpy.ndarray)が返される。

## グレースケール化

```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
BGR画像をグレースケール変換する。

第一引数：多次元配列  
第二引数：変更前の画像の色と、変更後の画像の色を示す定数

## エッジ抽出

```
edges = cv2.Canny(gray, 100, 400, apertureSize=3)
```
画素値の微分値を計算する。  
maxVal以上であれば、正しいエッジと判断する。  
minVal以下であれば、エッジではないと判断する。  

第一引数：エッジかどうかを判断する閾値(minVal)  
第二引数：エッジかどうかを判断する閾値(maxVal)  

一般に、数値が大きいほどエッジが検出されにくく、小さいほどエッジが検出されやすくなる。

## 二値画像の保存

```
cv2.imwrite('edges.png', edges)
```
多次元配列の情報をもとに、画像を保存する。

第一引数：保存先の画像ファイル名  
第二引数：多次元配列  
第三引数（任意）：リスト型にてパラメータを指定  

## 膨張処理

```
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
```
カーネルの作成。カーネルとは、入力画像の走査を行うために、各要素が0または1で構成された二次元の配列のこと。その中でも、アンカー（カーネルの原点）をスライドさせて、出力画像の画素を計算する。

第一引数：近傍とする形状
- cv2.MORPH_RECT: 長方形
- cv2.MORPH_CROSS: 十字
- cv2.MORPH_ELLIPSE: 楕円

第二引数：カーネルの大きさ

第三引数：アンカー部分。(-1, -1)の場合は、カーネルの中心

```
edges = cv2.dilate(edges, kernel)
```
膨張処理。カーネル内の画素値が1である画素が1つでも含まれている場合、出力画素の画素値を1にする。これによって、二値化する際の白の要素が増える。

第一引数：多次元配列  
第二引数：カーネル

## 二値画像から輪郭抽出

```
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```
二値画像から輪郭を抽出する。入力画像は、非0の画素を1とした二値画像である。

第一引数：入力画像  
第二引数：輪郭を検索する方法の指定
- cv2.RETR_EXTERNAL: 一番外側の輪郭のみ抽出する
- cv2.RETR_LIST: すべての輪郭を抽出するが、階層構造は作成しない
- cv2.RETR_CCOMP: すべての輪郭を抽出し、2階層の階層構造を作成する
- cv2.RETR_TREE: すべての輪郭を抽出し、ツリーで階層構造を作成する

第三引数：返り値の輪郭の点にオフセットを加算したい場合に指定する。
- cv2.CHAIN_APPROX_NONE
- cv2.CHAIN_APPROX_SIMPLE
- cv2.CHAIN_APPROX_TC89_L1
- cv2.CHAIN_APPROX_TC89_KCOS

## 面積によるフィルタリング
```
rects = []
for cnt, hrchy in zip(contours, hierarchy[0]):
    if cv2.contourArea(cnt) < 3000:
        continue  # 面積が小さいものは除く
    if hrchy[3] == -1:
       continue  # ルートノードは除く
    # 輪郭を囲む長方形を計算する。
    rect = cv2.minAreaRect(cnt)
    rect_points = cv2.boxPoints(rect).astype(int)
    rects.append(rect_points)
```
抽出した輪郭のうち、輪郭の面積が一定以上のものを選択する。

```
if cv2.contourArea(cnt) < 3000:
    continue  # 面積が小さいものは除く
```
各輪郭の面積を計算する。面積が小さい場合は、矩形領域として扱わないため、処理をスキップする。

```
if hrchy[3] == -1:
   continue  # ルートノードは除く
```
輪郭の階層構造を表すhierarchyリストのうち、ルートノードは矩形領域として扱わないため、処理をスキップする。

```
rect = cv2.minAreaRect(cnt)
```
最小外接矩形を計算する。検出した輪郭に対して、それらを囲む長方形の面積が最小になるような長方形を計算する。

```
rect_points = cv2.boxPoints(rect).astype(int)
rects.append(rect_points)
```
最小外接矩形を囲む四角形の頂点座標を計算する。計算された座標をリストに追加し、最終的に全ての矩形領域を保持するリストであるrectsに格納している。

## 描画

```
for i, rect in enumerate(rects):
    color = np.random.randint(0, 255, 3).tolist()
    cv2.drawContours(img, rects, i, color, 2)
    cv2.putText(img, str(i), tuple(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
    
    print('rect(%d):\n' %i, rect)

cv2.imwrite('img.png', img)
```
領域を読み取った画像を描画する。

```
color = np.random.randint(0, 255, 3).tolist()
```
ランダムな色を生成する。

第一引数：最小の値  
第二引数：指定した数値-1を最大の値  
第三引数：出力される配列のshapeを指定  
.tolistメソッドを使用し、リストに変換。

```
cv2.drawContours(img, rects, i, color, 2)
```
輪郭の抽出情報をもとに、輪郭の描画を行う。

第一引数：多次元配列  
第二引数：輪郭を形成する画素情報  
第三引数：輪郭を形成する画素のインデックス番号の指定。0を指定すると、1番目の輪郭を形成する画素のみを描画する。  
第四引数：輪郭を形成する画素の色の指定。BGR指定。  
第五引数（任意）：輪郭を形成するg阿蘇の大きさを指定。デフォルトでは1。  

```
print('rect(%d):\n' %i, rect)
```
抽出した矩形領域の各頂点の位置情報を、左下、左上、右上、右下の順に表示する。rect()の()の番号が、img.pngの画像で割り振られた領域の番号と一致する。

```
cv2.imwrite('img.png', img)
```
画像の保存を行う。

第一引数：保存先の画像ファイル名  
第二引数：多次元配列  

# URL
- [OpenCVで使われるcvtcolorとは?cvtcolorの活用例を徹底紹介](https://kuroro.blog/python/7IFCPLA4DzV8nUTchKsb/)

- [cv2.Canny(): Canny法によるエッジ検出の調整をいい感じにする](https://qiita.com/Takarasawa_/items/1556bf8e0513dca34a19)

- [cv2.Canny(): Canny法によるエッジ検出の自動化](https://qiita.com/kotai2003/items/662c33c15915f2a8517e)

- [OpenCVで使われるimwriteとは?imwriteの定義から使用例をご紹介](https://kuroro.blog/python/i0tNE1Mp8aEz8Z7n6Ggg/)

- [OpenCV – モルフォロジー演算 (膨張、収縮、オープニング、クロージング)](https://pystyle.info/opencv-morpology-operation/#outline__3_1)

- [モルフォロジー変換](https://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)

- [OpenCV – findContours で画像から輪郭を抽出する方法](https://pystyle.info/opencv-find-contours/#outline__4_1)

- [OpenCVで使われるdrawContoursって?具体的な活用法を徹底解説!?](https://kuroro.blog/python/xaw33ckABzGLiHDFWC3m/)
