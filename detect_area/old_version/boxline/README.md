# コマンド

実行コマンド。python では bash: python: command not found のエラーが出る。  
python3 を pip install しているため。  
過去バージョンのものについては、old_version/boxline ディレクトリにまとめている。

```
$ python3 export_array_boxline6.py
```

# 修正事項

- 矩形領域との重複認識
~~先述の通り、矩形領域の辺を水平線として認識しないよう、例外処理を加える必要がある。現在調整中。~~
→ export_array_boxline7.py (8月7日現在における最新バージョンなので、root ディレクトリの export_array_boxline のこと)で解決。検出した水平線の中点のx, y座標を矩形領域の頂点のx, y座標と比較する。許容誤差内であれば矩形領域の上下辺とみなし、リストから除外することで問題を解決。

- エッジ検出を行った場合の2本の重複認識
~~export_array_boxline3.py では、前処理としてエッジ検出を施している。このファイルを実行して生成される img2.png を見ると、水平線の上下にそれぞれ水平線が検出されており、最低2本の水平線が一本の線から検出される。本数は、先述の矩形領域との重複でさらに多くなることもある。~~  
→ export_array_boxline6.py で解決。各座標を平均し、別のリストに保持することで問題を解消。

# 解決策

- [x]条件分岐の見直し
見通しが立たないが、条件を見直すことは必須。一部の検出は除外できているため、できているものとできていないものを洗い出すとよいかもしれない。

- [x]2本の水平線のy座標を平均する
エッジ検出の上下に線1本につき2本の水平線が検出される問題については、それぞれの水平線の両端のy座標を平均し、更新するとよいと思われる。これによって、水平線2本の間にy座標が更新される。


# 各関数の詳細

矩形領域のものは省略する。


## エッジ検出

```
med_val = np.median(img_bw)
sigma = 0.33
min_val = int(max(0, (1.0 - sigma) * med_val))
max_val = int(max(255, (1.0 + sigma) * med_val))
edges = cv2.Canny(img_bw, threshold1 = min_val, threshold2 = max_val)
```

エッジ検出。後述するハフ変換のため。矩形領域機能にエッジ検出を施すと、最小外接矩形が認識されてしまうため、下線部認識のみで行う前処理である。  
今回は、二値画像の img_bw に対して、NumPy の np.median を使用することで、要素の中央値を求めている。これによって、本来2つの閾値を人が決定しなければならないところを、自動化している。  

参考：[cv2.Canny(): Canny 法によるエッジ検出の自動化](https://qiita.com/kotai2003/items/662c33c15915f2a8517e)

## ハフ変換

```
rect_sorted_memory = np.array(rect_sorted_memory)

unique_horizontal_lines = []

height, width, _ = img.shape
min_length = width * 0.1

lines = []
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=int(retval), minLineLength=min_length, maxLineGap=1)

```

rect_sorted_memory に矩形領域の各頂点の座標を保存しておく。  
空のリスト unique_horizontal_lines を後々のため用意する。  
img.shape で入力画像の高さと幅を求め、検知する直線の最低限の長さを幅の10%と決める。  
空のリスト lines に、ハフ変換の結果得られた直線の左端のx座標とy座標、右端のx座標とy座標の順で代入する。  

第一引数：入力画像
第二引数：rho(ロー)の値。 xcosΘ + ysinΘ = ρ の ρ にあたる。float 型。  
第三引数：theta（シータ）の値。 xcosΘ + ysinΘ = ρ の Θ にあたる。float 型。  
第四引数：閾値。今回は、大津の二値化で得た閾値を int に変換して利用している。  
第五引数：指定した数値以下の線は検出しないとする値。float 型。任意。  
第六引数：点と点の間隔をどれだけ許容するか。大きいほど直線と認識する。float 型。任意。  

## 水平線のみの検出に限定する

```
if lines is None:
    print('No straight lines detected')
    sys.exit()
else:
    for line in lines:
        tl_x, tl_y, br_x, br_y = line[0]
        # 傾き3px以内で検出対象に
        if abs(tl_y - br_y) < 3:
            line_list.append((tl_x, tl_y, br_x, br_y))
            
    line_list = sorted(line_list, key=lambda x: x[0])
```

新たに line_list という空のリストを用意し、横の直線のみに限定して各座標の値を代入する。 　
line に代入した直線の座標のうち、左端のy座標と右端のy座標の差の絶対値をとることで、線の傾きを求める。今回は、この値が3px以下であるもののみ検出することとしている。  
最後に、line_list に代入された値を、左端のx座標を基準にリストごとソートする。このとき、line_list リストには水平線の数だけリストが代入されている状態。  
線を検知しなかった場合は、その旨のメッセージを表示して終了する。

## 上下2本の線の座標を平均し、別のリストに代入

```
line_mean_list = []
    # line_listから処理済みの要素を削除するためにコピーを作る
    line_list_copy = line_list.copy()
    
    # line_list_copyが空になるまでループ
    while line_list_copy:
        # line_list_copyから最初の要素を取り出す
        left_x1, left_y1, right_x1, right_y1 = line_list_copy.pop(0)
        tmp_list = [(left_x1, left_y1, right_x1, right_y1)]
        
        # line_list_copyから他の要素を順番に取り出す
        for left_x2, left_y2, right_x2, right_y2 in line_list_copy:
            # エラーの範囲内であれば、一時保存リストに追加する
            if abs(left_y1 - left_y2) <= error and abs(left_x1 - left_x2) <= error:
                tmp_list.append((left_x2, left_y2, right_x2, right_y2))
```

エッジ検出を施すと、1本の線に対して、上下で2本の水平線が検出されてしまうという現象が発生した。これを解消するため、上下の線の座標を平均することとした。  
新たに line_mean_list という空のリストを用意し、左端のx, y座標が近傍する場合は、各座標の値を平均し、代入する。  
line list をコピーしたリスト line_list_copy から最初の要素をポップし、ポップした要素の要素、すなわち各座標の値を順番に取り出し、左端のy座標と右端のy座標の差の絶対値がエラーよりも小さい場合のみ、一時保存リストに追加する。

## 一時保存リストから各座標ごとに平均値を求め、別のリストに代入

```
        # 一時保存リストから平均値を計算する
        mean_left_x, mean_left_y, mean_right_x, mean_right_y = [np.mean([x[i] for x in tmp_list]) for i in range(4)]
        new_line = (int(mean_left_x), int(mean_left_y), int(mean_right_x), int(mean_right_y))
        line_mean_list.append(new_line)
        
        # 一時保存リストに含まれる要素をline_list_copyから削除する
        for line in tmp_list:
            if line in line_list_copy:
                line_list_copy.remove(line)

    line_nparray = np.array(line_mean_list)
```

mean_left_x, mean_left_y, mean_right_x, mean_right_y に、それぞれ 左端のx座標, 左端のy座標, 右端のx座標, 右端のy座標の平均値を代入し、new_line リストに int 型でそれぞれ代入する。  
line_list_copy に tmp_list の要素が含まれている場合、line_list_copy から該当要素を削除後、繰り返しを進める。  
ループが終了した後、まとめて ndarray 型へと変換する。

## 水平線が矩形領域の一部である場合、配列から排除する。

```
    for i, line in enumerate(line_nparray):
        for j in range(rect_sorted_memory.shape[0]):
            is_underline = True
            line_mid_x = (line_nparray[i][0] + line_nparray[i][2]) / 2
            line_mid_y = (line_nparray[i][1] + line_nparray[i][3]) / 2
            
            # 水平線の左端のx座標が、矩形領域の各辺の許容誤差内にあるかどうかを確認する
            if ( ( (rect_sorted_memory[j][0][0] - error <= line_mid_x <= rect_sorted_memory[j][0][0] + error)
                and (rect_sorted_memory[j][3][0] - error <= line_mid_x <= rect_sorted_memory[j][3][0] + error) )
                or ( (rect_sorted_memory[j][1][0] - error <= line_mid_x <= rect_sorted_memory[j][1][0] + error)
                and (rect_sorted_memory[j][2][0] - error <= line_mid_x <= rect_sorted_memory[j][2][0] + error) ) ):
                

                # 水平線の左端のy座標が、矩形領域の各辺の許容誤差内にあるかどうかを確認する
                if ( ( (rect_sorted_memory[j][0][1] - error <= line_mid_y <= rect_sorted_memory[j][0][1] + error)
                    and (rect_sorted_memory[j][3][1] - error <= line_mid_y <= rect_sorted_memory[j][3][1] + error) )
                    or ( (rect_sorted_memory[j][1][1] - error <= line_mid_y <= rect_sorted_memory[j][1][1] + error)
                    and (rect_sorted_memory[j][2][1] - error <= line_mid_y <= rect_sorted_memory[j][2][1] + error) ) ):
                    is_underline = False
                    
            # 重複フラグがTrueであれば、水平線は重複していないと判断し、リストに追加する
            if is_underline:
                unique_horizontal_lines = line_nparray.tolist()
                unique_horizontal_nparray = np.append(unique_horizontal_lines, line_nparray, axis=0)
```

要修正ポイント。現在は、水平線の中点の座標が矩形領域上にある場合、除外するという操作を行う予定である。

### 修正後 (export_array_boxline7.py)

```
    for i in range(rect_sorted_memory.shape[0]):
        for j, line in enumerate(line_nparray):
            is_underline = True
            line_mid_x = (line_nparray[j][0] + line_nparray[j][2]) / 2
            line_mid_y = (line_nparray[j][1] + line_nparray[j][3]) / 2
            
            # 水平線の中点の座標を確認。矩形の上辺について、x座標は両端の間で、かつy座標が誤差範囲か
            if ( (rect_sorted_memory[i][0][0] - rect_error <= line_mid_x <= rect_sorted_memory[i][3][0] + rect_error)
                and ( (rect_sorted_memory[i][0][1] - rect_error <= line_mid_y <= rect_sorted_memory[i][0][1] + rect_error)
                or (rect_sorted_memory[i][3][1] - rect_error <= line_mid_y <= rect_sorted_memory[i][3][1] + rect_error) ) ):
                overlap_index.append(j)

            # 水平線の中点の座標を確認。矩形の下辺について、x座標は両端の間で、かつy座標が誤差範囲か
            if ( (rect_sorted_memory[i][1][0] - rect_error <= line_mid_x <= rect_sorted_memory[i][2][0] + rect_error)
                and ( (rect_sorted_memory[i][1][1] - rect_error <= line_mid_y <= rect_sorted_memory[i][1][1] + rect_error)
                or (rect_sorted_memory[i][2][1] - rect_error <= line_mid_y <= rect_sorted_memory[i][2][1] + rect_error) ) ):
                overlap_index.append(j)
    
    # 重複する水平線のインデックスを参照し、ndarray 配列から削除               
    unique_horizontal_nparray = np.delete(line_nparray, overlap_index, 0)
```

この修正により、矩形領域の上下辺を水平線として検知しなくなった。  
水平線の中点のx, y座標を計算し、矩形領域の各頂点のx, y座標を参照。矩形領域の辺とみなす誤差 rect_error を考慮し、範囲内であれば検知した水平線のリストから除外する。  

## 水平線の描画と各座標の出力
```
# 矩形領域と重複しない水平線の座標を表示する
for i, line in enumerate(unique_horizontal_lines):
    x1, y1, x2, y2 = unique_horizontal_lines[i]
    cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(img2, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    print('line(%d):' %i, unique_horizontal_lines[i])

cv2.imwrite('img2.png', img2)
```

重複しない水平線を代入したリストの要素数だけ、画像に緑色の線を描画し、座標をターミナル上に出力する。ターミナル上の line(数字) の数字と、画像に描画された数字は一致する。  
出力画像は、img2.png という名前で保存される。


# 参考文献

-[【Python OpenCV】画像 から 横線 のみ検出する(おまけで縦線も)](https://qiita.com/youichi_io/items/6b08519137819354a435#%E4%BD%BF%E7%94%A8%E3%83%A9%E3%82%A4%E3%83%96%E3%83%A9%E3%83%AA)

-[要素の中央値を計算するNumPyのmedian関数の使い方](https://deepage.net/features/numpy-median.html)

-[while文を使ったリストのループ処理](https://www.python.jp/train/list/list_loop.html)

-[OpenCVで使われるHoughLinesPとは?定義から実用例を徹底解説!?](https://kuroro.blog/python/nNffXtmWKE3lEa6bbbSw/)

-[Pythonでリストのサイズ（要素数）を取得](https://note.nkmk.me/python-list-len/)

-[NumPy配列ndarrayの次元数、形状、サイズ（全要素数）を取得](https://note.nkmk.me/python-numpy-ndarray-ndim-shape-size/)

-[OpenCVで画像を読み込んだときのsizeとshapeの意味](https://qiita.com/hiratake_0108/items/d193911c5d30700272a3)