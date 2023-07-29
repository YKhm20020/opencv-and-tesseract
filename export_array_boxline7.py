import os
import cv2
import numpy as np

# 頂点を左上、左下、右下、右上の順序に並び替える関数
def sort_points(points):
    # x座標とy座標の和が最小のものが左上
    tl = min(points, key=lambda x: x[0] + x[1])
    # x座標とy座標の差が最小のものが右上
    tr = min(points, key=lambda x: x[0] - x[1])
    # x座標とy座標の和が最大のものが右下
    br = max(points, key=lambda x: x[0] + x[1])
    # x座標とy座標の差が最大のものが左下
    bl = max(points, key=lambda x: x[0] - x[1])
    # 順序に従ってリストにする
    return [tl, tr, br, bl]

# ディレクトリ作成、入力画像の決定と読み取り
results_path = './results'
if not os.path.exists(results_path):
    os.mkdir(results_path)

input_image = './sample/sample4.jpg'

img = cv2.imread(input_image)
img2 = cv2.imread(input_image)

# BGR -> グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
# TOZERO_BINARY で直線をひとつ多く検出したことを確認。他サンプルと比較必須。
retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite('result1.png', img_bw) # 確認用

# 矩形領域と下線部検出で処理を分けるか検討。ガウシアンフィルタ適用の是非など。
# img_gray_line = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray_line2 = cv2.bitwise_not(img_gray_line)

# 膨張処理
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
img_bw = cv2.dilate(img_bw, kernel, iterations = 1)
cv2.imwrite('result2.png', img_bw) # 確認用

# Canny 法によるエッジ検出（下線部検出のみ）
med_val = np.median(img_bw)
sigma = 0.33
min_val = int(max(0, (1.0 - sigma) * med_val))
max_val = int(max(255, (1.0 + sigma) * med_val))
edges = cv2.Canny(img_bw, threshold1 = min_val, threshold2 = max_val)
cv2.imwrite('result3.png', edges) # 確認用

# 以下、矩形領域検出
# 輪郭抽出
contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 面積でフィルタリング
rects = []
for cnt, hrchy in zip(contours, hierarchy[0]): 
    if cv2.contourArea(cnt) < 3000:
        continue  # 面積が小さいものを除外
    if hrchy[3] == -1:
        continue  # ルートノードを除外
   
    # 輪郭を囲む長方形を計算する。
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    
    # 縦横の線の長さを比較し、どちらがの線の長さが極端に短い場合は除外
    if min(w, h) < 10:
        continue
    
    rect_points = cv2.boxPoints(rect).astype(int)
    rects.append(rect_points)

# x-y 順でソート
rects = sorted(rects, key=lambda x: (x[0][1], x[0][0]))

rect_sorted_memory = []

# 描画する。
i = 0
for i, rect in enumerate(rects):
    # 頂点を左上、左下、右下、右上の順序に並び替える
    rect_sorted = np.array(sort_points(rect))
    
    rect_sorted_memory.append(rect_sorted)
    
    color = np.random.randint(0, 255, 3).tolist()
    cv2.drawContours(img, rects, i, color, 2)
    cv2.putText(img, str(i), tuple(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    print('rect(%d):\n' %i, rect_sorted)
    
print()
cv2.imwrite('img.png', img)


# 以下は下線部認識
rect_sorted_memory = np.array(rect_sorted_memory)

height, width, _ = img.shape
min_length = width * 0.1

# ハフ変換による直線検出
lines = []
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=int(retval), minLineLength=min_length, maxLineGap=1)

line_list = []
error = 10 # 検知した直線を矩形の一部とみなす誤差
if lines is not None:
    for line in lines:
        tl_x, tl_y, br_x, br_y = line[0]
        # 傾き3px以内で検出対象に
        if abs(tl_y - br_y) < 3:
            tl_x = line[0][0]
            tl_y = line[0][1]
            br_x = line[0][2]
            br_y = line[0][3]
            line = (tl_x, tl_y, br_x, br_y)
            line_list.append(line)
            
    line_list = sorted(line_list, key=lambda x: x[0])
    
    line_mean_list = []
    # line_listから処理済みの要素を削除するためにコピーを作る
    line_list_copy = line_list.copy()
    
    # line_list_copy が空になるまでループ
    while line_list_copy:
        # line_list_copy から最初の要素を取り出す
        left_x1, left_y1, right_x1, right_y1 = line_list_copy.pop(0)
        tmp_list = [(left_x1, left_y1, right_x1, right_y1)]
        
        # line_list_copy から他の要素を順番に取り出す
        for left_x2, left_y2, right_x2, right_y2 in line_list_copy:
            # エラーの範囲内であれば、一時保存リストに追加する
            if abs(left_y1 - left_y2) <= error and abs(left_x1 - left_x2) <= error:
                tmp_list.append((left_x2, left_y2, right_x2, right_y2))
                
        # 一時保存リストから各座標ごとに平均値を計算する
        mean_left_x = np.mean([x[0] for x in tmp_list])
        mean_left_y = np.mean([x[1] for x in tmp_list])
        mean_right_x = np.mean([x[2] for x in tmp_list])
        mean_right_y = np.mean([x[3] for x in tmp_list])
        new_line = (int(mean_left_x), int(mean_left_y), int(mean_right_x), int(mean_right_y))
        line_mean_list.append(new_line)
        
        # 一時保存リストに含まれる要素をline_list_copyから削除する
        for line in tmp_list:
            if line in line_list_copy:
                line_list_copy.remove(line)

    line_nparray = np.array(line_mean_list)

    # 重複する水平線のインデックスを保存するリスト
    overlap_index = []
    i = 0
    
    for i in range(rect_sorted_memory.shape[0]):
        j = 0
        for j, line in enumerate(line_nparray):
            is_underline = True
            line_mid_x = (line_nparray[j][0] + line_nparray[j][2]) / 2
            line_mid_y = (line_nparray[j][1] + line_nparray[j][3]) / 2
            
            # 水平線の中点の座標を確認。矩形の上辺について、x座標は両端の間で、かつy座標が誤差範囲か
            if ( (rect_sorted_memory[i][0][0] - error <= line_mid_x <= rect_sorted_memory[i][3][0] + error)
                and ( (rect_sorted_memory[i][0][1] - error <= line_mid_y <= rect_sorted_memory[i][0][1] + error)
                or (rect_sorted_memory[i][3][1] - error <= line_mid_y <= rect_sorted_memory[i][3][1] + error) ) ):
                overlap_index.append(j)

            # 水平線の中点の座標を確認。矩形の下辺について、x座標は両端の間で、かつy座標が誤差範囲か
            if ( (rect_sorted_memory[i][1][0] - error <= line_mid_x <= rect_sorted_memory[i][2][0] + error)
                and ( (rect_sorted_memory[i][1][1] - error <= line_mid_y <= rect_sorted_memory[i][1][1] + error)
                or (rect_sorted_memory[i][2][1] - error <= line_mid_y <= rect_sorted_memory[i][2][1] + error) ) ):
                overlap_index.append(j)
    
    # 重複する水平線のインデックスを参照し、ndarray 配列から削除               
    unique_horizontal_nparray = np.delete(line_nparray, overlap_index, 0)

else:
    print('No straight lines detected')

# 矩形領域と重複しない水平線の座標を表示する
i = 0
if unique_horizontal_nparray.shape[0] == 0:
    print('Does not exist underline')
else:
    for i, line in enumerate(unique_horizontal_nparray):
        x1, y1, x2, y2 = unique_horizontal_nparray[i]
        cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img2, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print('line(%d):' %i, unique_horizontal_nparray[i])


    # # 全ての頂点を一次元配列にする
    # all_points = np.concatenate(rects)

    # # x座標とy座標を別々に取り出す
    # x_coords = all_points[:, 0]
    # y_coords = all_points[:, 1]

    # # 最小値と最大値を求める
    # min_x = np.min(x_coords)
    # max_x = np.max(x_coords)
    # min_y = np.min(y_coords)
    # max_y = np.max(y_coords)
        
    # print("min_x: %d" % min_x)
    # print("max_x: %d" % max_x)
    # print("min_y: %d" % min_y)
    # print("max_y: %d" % max_y)

    cv2.imwrite('img2.png', img2)
