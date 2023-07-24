import os
import cv2
import numpy as np
import math

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

input_image = './sample/sample2.jpg'

img = cv2.imread(input_image)

# BGR -> グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite('result1.png', img_bw) # 確認用

# 膨張処理
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
img_bw = cv2.dilate(img_bw, kernel, iterations = 1)
cv2.imwrite('result2.png', img_bw) # 確認用

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

# 描画する。
for i, rect in enumerate(rects):
    # 頂点を左上、左下、右下、右上の順序に並び替える
    rect_sorted = np.array(sort_points(rect))
    
    color = np.random.randint(0, 255, 3).tolist()
    cv2.drawContours(img, rects, i, color, 2)
    cv2.putText(img, str(i), tuple(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    print('rect(%d):\n' %i, rect_sorted)

# 重複していない水平線の座標を格納するリスト
unique_horizontal_lines = []

# 平行な水平線のみ取り扱うため、ふるいにかける
height, width, _ = img.shape
min_length = width * 0.2

lines = []
lines = cv2.HoughLinesP(img_bw, rho=1, theta=np.pi/360, threshold=100, minLineLength=min_length, maxLineGap=1)

line_list = []
error = 10 # 矩形の直線とみなす許容誤差
if lines is not None:
    for line in lines:
        tl_x, tl_y, br_x, br_y = line[0]
        is_underline = False
        # 傾きを3px以内と判断
        if abs(tl_y - br_y) < 1:
            whiteline = 1
            lineadd_img = cv2.line(img_bw, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 255), whiteline)
            tl_x = line[0][0]
            tl_y = line[0][1]
            br_x = line[0][2]
            br_y = line[0][3]
            line = (tl_x, tl_y, br_x, br_y)
            
            line_list.append(line)
            line_nparray = np.array(line_list)

    for i, line in enumerate(line_nparray):
        # 水平線の両端のx座標が、矩形領域の左上と右上、または左下と右下のx座標の間にあるかどうかを確認する
        if ((abs(rect_sorted[0][0] - line_nparray[i][0] <= error) or (abs(rect_sorted[3][0] - line_nparray[i][1] <= error)) ) or (abs(rect_sorted[0][0] or line_nparray[i][2] <= error) or (abs(rect_sorted[3][0] - line_nparray[i][2] <= error)) ) ):
            # 水平線の両端のy座標が、矩形領域の左上と左下、または右上と右下のy座標と同じかどうかを確認する
            if ( (abs(rect_sorted[0][1] - line_nparray[i][1] <= error) and (abs(rect_sorted[0][1] - line_nparray[i][3] <= error)) ) or ( abs(rect_sorted[1][1] - line_nparray[i][1] <= error) or abs(rect_sorted[1][1] - line_nparray[i][3] <= error) ) ):
                # 両方の条件を満たす場合は、重複フラグをTrueにする
                is_underline = True
            
            
        # 重複フラグがTrueであれば、水平線は重複していないと判断し、リストに追加する
        if is_underline:
            unique_horizontal_lines += line_nparray.tolist()

unique_horizontal_nparray = np.array(unique_horizontal_lines)

# 矩形領域と重複しない水平線の座標を表示する
for i, line in enumerate(unique_horizontal_lines):
    x1, y1, x2, y2 = unique_horizontal_lines[i]
    # 線を描画する
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 数字を描画する
    cv2.putText(img, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    print('line(%d):' %i, unique_horizontal_lines[i])

    
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

cv2.imwrite('img.png', img)
