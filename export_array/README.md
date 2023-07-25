# コマンド

実行コマンド。python では bash: python: command not found のエラーが出る。  
python3 を pip install しているため。  
過去バージョンのものについては、old_version ディレクトリにまとめている。

```
$ python3 export_array.py
```

# 修正事項

1. 親領域と子領域の重複  
   完全な矩形領域でない場合は、大きさの異なる矩形領域の集合である。
   この場合、最小の矩形領域に加えて、それらの集合である 1 つの大きな矩形領域が別に抽出されてしまう。

~~→ 概ね解決？~~  
~~矩形領域取得の際に、面積の小さい矩形領域を複数含み、それらが内接する最小の矩形領域（外側にある大きな矩形領域のこと）も余計に取得されていた。~~
~~しかし、膨張処理に cv2.Canny でない方法を採用することで解決。~~
~~（sample_result3, 4 の 2 種の画像を参照）~~

→ 本格的に解決  
cv2.THRESH_BINARY_INV によって解決。同時に、大津の二値化によって閾値の自動決定も可能に。  

追記：
原因は、エッジ検出にあった可能性がある。各矩形領域が集まっている場合、矩形の内側と、矩形の集合した外側でそれぞれ矩形を検出してしまっていた可能性が高い。原理は、boxline の水平線2本検出と同様。

2. 領域取得の漏れ  
   特に面積の小さなもの（履歴書の学歴・職歴の月など）の取得が一部できていない場合がある。
   （未検証だが、面積が小さいことに加え、それらが隣接している場合に起こる傾向にある）
   また、文字を誤って極小領域として認識してしまう場合もある。
   sample_result_picture2.png では、上 11 個、下 48 個の枠を取得する想定だが、結果は上 13 個、下 44 個の枠が取得されている。
   これは、上部で写真を張る箇所に書かれている文字が誤って領域取得され、想定よりも領域の数が多く、
   下部で一部の欄（月の欄、例えば下から 4 行目の、48 番目の領域が本来は年と月の 2 つに分かれて取得されなければならない）

3. 値の補正

```
img_gray = cv2.blur(img_gray, (3,3))
img_bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 1101, 10)
```

~~上の 2 行の関数呼び出しの際、引数の値は自分で検証しながら適切な値を決定している。~~
~~ここを解決しなければ研究の意義がそのままなくなるため、特に修正が必要な箇所である。~~

→ 解決  
export_array の大津の二値化によって、閾値の自動決定に成功。

# 検証事項

1. 撮影環境  
   入力画像は撮影する環境に依存し、出力画像に大きく影響する。  
   よって、以下の条件を変更・検証して機能の実装を進めていくべきである。

- カメラとの距離
- 撮影場所の明るさ
- 撮影角度

2. 別のテンプレートによる入力  
   現在は履歴書だけが入力だが、他のテンプレートでどのように動作するかは調査する必要がある。 具体的には、それぞれでテンプレートそのままの画像と、テンプレートを印刷してスマホで撮った画像の 2 種類が必要である。  
   さらに、枠のない入力画像（下線のみが引いてあるなど）の場合はどのように動作するかも調べる必要がある。

# 各バージョンごとの違い

今のところ、ファイル名の後ろの数が大きいほど抽出の精度は高い。場合によっては、前バージョンの関数を異なる方法で実装することも考えられる。

## 過去バージョン

### export_array

エッジ抽出に Canny を使用。精度が悪く、各矩形領域を内接する大きな矩形領域も抽出してしまう。

### export_array2

抽出した矩形領域の各頂点の座標(px)を表示。また、矩形領域ごとに出力した座標がわかりやすく番号付け。大きな矩形領域は排除できず。

### export_array3

閾値処理に threshold を使用。二値化の方法に THRESH_BINARY_INV を選択し、矩形領域を内接する大きな矩形領域を排除することに成功。

### export_array4

閾値処理に adaptiveThreshold を使用。二値化の閾値を領域ごとに決定することができるものの、近傍サイズによってはある入力画像に対して最適に領域取得ができるが、別の入力画像に対してはうまく領域が取得できないことを確認済みである。

### export_array5

カーネル作成に cv2.getStructuringElement を使用。モルフォジー変換を行う。

## 現行バージョン

### export_array6

cv2.threshold に cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU として閾値を決定する。cv2.THRESH_BINARY_INV により外側の余計な矩形領域の除外に、大津の二値化により、閾値を自動決定することに成功。しかし、cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))の 3 を 4 にすると精度が上がるものの、文字を誤って矩形領域として検出してしまうことがある問題がある。

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
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

BGR 画像をグレースケール変換する。

第一引数：入力画像（多次元配列）  
第二引数：変更前の画像の色と、変更後の画像の色を示す定数

## 画像の平滑化

```
img_gray = cv2.blur(img_gray, (3,3))
```

ブラー処理を行う。
カーネルサイズのうち、サイズ領域の平均値を中央の値（アンカー）に適応する。
これによって、画像をぼかすことができる。
検証したところ、履歴書の入力画像については、入力画像にブラー処理を適応してグレースケール化する場合も、グレースケール後の画像にブラー処理を適応する場合も、枠の取得結果に変化はなかった。

第一引数：入力画像（多次元配列）  
第二引数：カーネルサイズ

## エッジ抽出

```
edges = cv2.Canny(gray, 100, 400, apertureSize=3)
```

画素値の微分値を計算する。  
maxVal 以上であれば、正しいエッジと判断する。  
minVal 以下であれば、エッジではないと判断する。

第一引数：エッジかどうかを判断する閾値(minVal)  
第二引数：エッジかどうかを判断する閾値(maxVal)

一般に、第一引数の数値が大きいほどエッジが抽出されにくく、小さいほどエッジが抽出されやすくなる。

### export_array3 での変更後

```
retval, img_bw = cv2.threshold(img_gray, 150, 300, cv2.THRESH_BINARY_INV)
```

ある 1 つの閾値を決めて、2 値化を行う cv2.threshold に変更。  
この関数の特徴として、代入先に返り値として代入される閾値も含めた 2 つが必要である。

第一引数：入力画像（多次元配列）  
第二引数：閾値  
第三引数：閾値（最大）  
第四引数：二値化の方法

二値化の方法にはいくつか種類がある。

- cv2.THRESH_BINARY
- cv2.THRESH_BINARY_INV
- cv2.THRESH_TRUNC
- cv2.THRESH_TOZERO
- cv2.THRESH_TOZERO_INV
- cv2.THRESH_OTSU
- cv2.THRESH_TRIANGLE

この中でも、第三引数の閾値は、cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV の場合に、使用する。
各方法の詳細については下記リンク参照。  
[OpenCV – 画像処理の 2 値化の仕組みと cv2.threshold() の使い方](https://pystyle.info/opencv-image-binarization/)

### export_array4 での変更後

```
img_bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 1101, 10)
```

cv2.adaptiveThreshold に変更。二値化には全画素を同じ閾値で変換する大域的二値化と、領域ごとに異なる閾値で変換する適応的二値化の 2 つの方法がある。  
明るさが均一でない場合は、大域的二値化ではうまく二値化ができない場合がある。  
この変更によって領域取得の精度は向上したものの、近傍サイズは適宜自分で決定していることが明確な改善点である。詳しくは 修正事項.3 を参照。

第一引数：入力画像（多次元配列）  
第二引数：二値化後の輝度値  
第三引数：適応的閾値処理で使用するアルゴリズム  
第四引数：二値化の種類  
第五引数：閾値計算のための近傍サイズ  
第六引数：平均あるいは加重平均から引かれる値

第三引数のアルゴリズムには 2 種類ある。

- cv2.ADAPTIVE_THRESH_MEAN_C
- cv2.ADAPTIVE_THRESH_GAUSSIAN_C

また、第四引数の二値化の種類も 2 種類ある。

- cv2.THRESH_BINARY
- cv2.THRESH_BINARY_INV
  詳しくは下記ページを参照。  
  [【OpenCV/Python】adaptiveThreshold の処理アルゴリズム](https://imagingsolution.net/program/python/opencv-python/adaptivethreshold_algorithm/)

## 二値画像の保存

```
cv2.imwrite('edges.png', edges)
```

多次元配列の情報をもとに、画像を保存する。

第一引数：保存先の画像ファイル名  
第二引数：入力画像（多次元配列）  
第三引数（任意）：リスト型にてパラメータを指定

## 膨張処理

```
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
```

カーネルの作成。カーネルとは、入力画像の走査を行うために、各要素が 0 または 1 で構成された二次元の配列のこと。その中でも、アンカー（カーネルの原点）をスライドさせて、出力画像の画素を計算する。

第一引数：近傍とする形状

- cv2.MORPH_RECT: 長方形
- cv2.MORPH_CROSS: 十字
- cv2.MORPH_ELLIPSE: 楕円

第二引数：カーネルの大きさ  
第三引数：アンカー部分。(-1, -1)の場合は、カーネルの中心

## export_array3 以降での変更

```
kernel = np.ones((2,2), np.uint8)
```

カーネル作成の際に、np.ones 関数を用いた。 これによって、2×2 配列が形状として指定される。  
uint8 は NumPy におけるデータ型 dtype のひとつであり、符号なし 8 ビット整数型を指す。  
int8 でも取得領域に変化がなかったため、あまり影響がない箇所？

第一引数：配列の形状  
第二引数：データ型  
第三引数（任意）：初期値（'C'または'F'が入り、配列のデータの保存の仕方を指定する）

詳しくは下記ページ参照。

- [AdaptiveThreshold で照明環境が微妙な画像を二値化](https://opqrstuvcut.github.io/blog/p/adaptivethreshold%E3%81%A7%E7%85%A7%E6%98%8E%E7%92%B0%E5%A2%83%E3%81%8C%E5%BE%AE%E5%A6%99%E3%81%AA%E7%94%BB%E5%83%8F%E3%82%92%E4%BA%8C%E5%80%A4%E5%8C%96/)
- [要素が 1 の配列を生成する numpy.ones 関数の使い方](https://deepage.net/features/numpy-ones.html)

```
edges = cv2.dilate(edges, kernel)
```

膨張処理。カーネル内の画素値が 1 である画素が 1 つでも含まれている場合、出力画素の画素値を 1 にする。これによって、二値化する際の白の要素が増える。

第一引数：入力画像（多次元配列）  
第二引数：カーネル
第三引数（任意）：イテレーション（数が大きくなるとより膨張する）

## 二値画像から輪郭抽出

```
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

二値画像から輪郭を抽出する。入力画像は、非 0 の画素を 1 とした二値画像である。

第一引数：入力画像（多次元配列）  
第二引数：輪郭を検索する方法の指定

- cv2.RETR_EXTERNAL: 一番外側の輪郭のみ抽出する
- cv2.RETR_LIST: すべての輪郭を抽出するが、階層構造は作成しない
- cv2.RETR_CCOMP: すべての輪郭を抽出し、2 階層の階層構造を作成する
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

輪郭の階層構造を表す hierarchy リストのうち、ルートノードは矩形領域として扱わないため、処理をスキップする。

```
rect = cv2.minAreaRect(cnt)
(x, y), (w, h), angle = rect
    
# 縦横の線の長さを比較し、どちらがの線の長さが極端に短い場合は除外
if min(w, h) < 10:
   continue
```

最小外接矩形を計算する。抽出した輪郭に対して、それらを囲む長方形の面積が最小になるような長方形を計算する。cv2.minAreaRect では [ (x座標,y座標), (幅, 高さ), 角度 ] のリストに各値が格納されている。その中で、幅と高さどちらかが極端に短いものは、輪郭検出の対象外としている。

```
rect_points = cv2.boxPoints(rect).astype(int)
rects.append(rect_points)
```

最小外接矩形を囲む四角形の頂点座標を計算する。計算された座標をリストに追加し、最終的に全ての矩形領域を保持するリストである rects に格納している。cv2.boxPoint で4点座標にし、わかりやすくしている。

## 描画

```
for i, rect in enumerate(rects):
    color = np.random.randint(0, 255, 3).tolist()
    cv2.drawContours(img, rects, i, color, 2)
    cv2.putText(img, str(i), tuple(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

    print('rect(%d):\n' %i, rect)

cv2.imwrite('img.png', img)
```

領域を読み取った画像を描画する。枠ごとに 0 から数字を割り振る。

```
color = np.random.randint(0, 255, 3).tolist()
```

ランダムな色を生成する。出力画像に枠を見やすく表示するため。

第一引数：最小の値  
第二引数：指定した数値-1 を最大の値  
第三引数：出力される配列の shape を指定  
.tolist メソッドを使用し、リストに変換。

```
cv2.drawContours(img, rects, i, color, 2)
```

輪郭の抽出情報をもとに、輪郭の描画を行う。

第一引数：入力画像（多次元配列）  
第二引数：輪郭を形成する画素情報  
第三引数：輪郭を形成する画素のインデックス番号の指定。0 を指定すると、1 番目の輪郭を形成する画素のみを描画する。  
第四引数：輪郭を形成する画素の色の指定。BGR 指定。  
第五引数（任意）：輪郭を形成する g 阿蘇の大きさを指定。デフォルトでは 1。

```
print('rect(%d):\n' %i, rect)
```

抽出した矩形領域の各頂点の位置情報を、左下、左上、右上、右下の順に表示する。rect()の()の番号が、img.png の画像で割り振られた領域の番号と一致する。

```
cv2.imwrite('img.png', img)
```

画像の保存を行う。

第一引数：保存先の画像ファイル名  
第二引数：入力画像（多次元配列）

# 参考文献

- [【Python】表の位置情報をOpenCVで検出する！](https://proglearn.com/2020/12/25/python%E8%A1%A8%E3%81%AE%E4%BD%8D%E7%BD%AE%E6%83%85%E5%A0%B1%E3%82%92opencv%E3%81%A7%E8%AA%AD%E3%81%BF%E5%8F%96%E3%82%8B%EF%BC%81/)

- [OpenCV で使われる cvtcolor とは?cvtcolor の活用例を徹底紹介](https://kuroro.blog/python/7IFCPLA4DzV8nUTchKsb/)

- [OpenCV – 画像処理の 2 値化の仕組みと cv2.threshold() の使い方](https://pystyle.info/opencv-image-binarization/)

- [OpenCV で使われる imwrite とは?imwrite の定義から使用例をご紹介](https://kuroro.blog/python/i0tNE1Mp8aEz8Z7n6Ggg/)

- [OpenCV – モルフォロジー演算 (膨張、収縮、オープニング、クロージング)](https://pystyle.info/opencv-morpology-operation/#outline__3_1)

- [モルフォロジー変換](https://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)

- [OpenCV – findContours で画像から輪郭を抽出する方法](https://pystyle.info/opencv-find-contours/#outline__4_1)

- [OpenCV で使われる drawContours って?具体的な活用法を徹底解説!?](https://kuroro.blog/python/xaw33ckABzGLiHDFWC3m/)
  
  
  
以降、boxline（下線部検出）

- [OpenCVで直線の検出](https://qiita.com/tifa2chan/items/d2b6c476d9f527785414#%E3%81%BE%E3%81%A8%E3%82%81)

- [cv2.Canny(): Canny 法によるエッジ検出の調整をいい感じにする](https://qiita.com/Takarasawa_/items/1556bf8e0513dca34a19)

- [cv2.Canny(): Canny 法によるエッジ検出の自動化](https://qiita.com/kotai2003/items/662c33c15915f2a8517e)

