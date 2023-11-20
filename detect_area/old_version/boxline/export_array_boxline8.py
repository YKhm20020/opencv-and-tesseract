import os
import sys
import cv2
import numpy as np
# 以下、データ出力用
import json
import csv

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
dir = ['./results', './data/txt', './data/json', './data/npy', './data/csv']
results_path, data_txt_path, data_json_path, data_npy_path, data_csv_path = dir

os.makedirs(results_path, exist_ok = True)
os.makedirs(data_txt_path, exist_ok = True)
os.makedirs(data_json_path, exist_ok = True)
os.makedirs(data_npy_path, exist_ok = True)
os.makedirs(data_csv_path, exist_ok = True)

input_image = './sample/sample.jpg'
#input_image = './sample/P/02稟議書_/A281新卒者採用稟議書.png'
#input_image = './sample/P/02稟議書_/A282広告出稿稟議書.png'
#input_image = './sample/P/02稟議書_/A321稟議書.png'
#input_image = './sample/P/02稟議書_/A438安全衛生推進者選任稟議書.png'
#input_image = './sample/P/02稟議書_/A481広告出稿稟議書.png'
#input_image = './sample/P/18作業報告書_/B090入庫報告書.png'
#input_image = './sample/P/26休暇届_/A089夏季休暇届.png'

img = cv2.imread(input_image)
img_underline = cv2.imread(input_image)

# BGR -> グレースケール, Gaussian フィルタ
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 2)

# 第四引数が cv2.THRESH_TOZERO_INV で直線をひとつ多く検出したことを確認。他サンプルと比較必須。
#retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)

cv2.imwrite(f'{results_path}/result1_gray.png', img_bw) # 確認用

# 膨張処理
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
img_bw = cv2.dilate(img_bw, kernel, iterations = 1)
cv2.imwrite(f'{results_path}/result2_bw.png', img_bw) # 確認用

# Canny 法によるエッジ検出（下線部検出のみ）
med_val = np.median(img_bw)
sigma = 0.33
min_val = int(max(0, (1.0 - sigma) * med_val))
max_val = int(max(255, (1.0 + sigma) * med_val))
edges = cv2.Canny(img_bw, threshold1 = min_val, threshold2 = max_val)
print(min_val, max_val)
cv2.imwrite(f'{results_path}/result3_edges.png', edges) # 確認用

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
    
    # 縦横の線のうち、どちらがの線の長さが極端に短い場合は除外
    if min(w, h) < 10:
        continue
    
    rect_points = cv2.boxPoints(rect).astype(int)
    rects.append(rect_points)

# x-y 順でソート
rects = sorted(rects, key=lambda x: (x[0][1], x[0][0]))

rect_sorted_memory = []

# 描画する。
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

rect_sorted_memory = np.array(rect_sorted_memory)
rect_sorted_list = rect_sorted_memory.tolist()

# .txt, .json, .csv ファイルで矩形領域の座標をエクスポート
with open(f'{data_txt_path}/rects_data.txt', 'w') as f:
    json.dump(rect_sorted_list, f)

with open(f'{data_json_path}/rects_data.json', 'w') as f:
    json.dump(rect_sorted_list, f)
    
with open(f'{data_csv_path}/rects_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(rect_sorted_list)



# 以下は下線部認識

height, width, _ = img.shape
min_length = width * 0.1

# ハフ変換による直線検出
lines = []
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=int(retval), minLineLength=min_length, maxLineGap=1)

line_list = []
same_line_error = 10 # 上下に生成される直線を同一のものと捉える誤差

if lines is None:
    print('No straight lines are detected')
    sys.exit()
else:
    for line in lines:
        tl_x, tl_y, br_x, br_y = line[0]
        # 傾き3px以内で検出対象に
        if abs(tl_y - br_y) < 10:
            line_list.append((tl_x, tl_y, br_x, br_y))
            
    line_list = sorted(line_list, key=lambda x: x[0])
    
    line_mean_list = []
    # line_list から処理済みの要素を削除するためにコピーを作る
    line_list_copy = line_list.copy()
    
    # line_list_copy が空になるまでループ
    while line_list_copy:
        # line_list_copy から最初の要素を取り出す
        left_x1, left_y1, right_x1, right_y1 = line_list_copy.pop(0)
        tmp_list = [(left_x1, left_y1, right_x1, right_y1)]
        
        # line_list_copy から他の要素を順番に取り出す
        for left_x2, left_y2, right_x2, right_y2 in line_list_copy:
            # 誤差の範囲内であれば、一時保存リストに追加する
            if abs(left_y1 - left_y2) <= same_line_error and abs(left_x1 - left_x2) <= same_line_error:
                tmp_list.append((left_x2, left_y2, right_x2, right_y2))
                
        # 一時保存リストから各座標ごとに平均値を計算する
        mean_left_x, mean_left_y, mean_right_x, mean_right_y = [np.mean([x[i] for x in tmp_list]) for i in range(4)]
        new_line = (int(mean_left_x), int(mean_left_y), int(mean_right_x), int(mean_right_y))
        line_mean_list.append(new_line)
        
        # 一時保存リストに含まれる要素を line_list_copy から削除する
        for line in tmp_list:
            if line in line_list_copy:
                line_list_copy.remove(line)

    line_nparray = np.array(line_mean_list)

    # 重複する水平線のインデックスを保存するリスト
    overlap_index = []
    rect_error = 10 # 検知した直線を矩形の一部と捉える誤差
    
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

    # 矩形領域と重複しない水平線の座標を表示する
    if unique_horizontal_nparray.shape[0] == 0:
        print('Does not exist underline')
    else:
        for i, line in enumerate(unique_horizontal_nparray):
            x1, y1, x2, y2 = unique_horizontal_nparray[i]
            cv2.line(img_underline, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_underline, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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

    cv2.imwrite('img_underline.png', img_underline)
    
    unique_horizontal_list = unique_horizontal_nparray.tolist()
    
    # .txt, .json, .csv ファイルで下線の座標をエクスポート
    with open(f'{data_txt_path}/underlines_data.txt', 'w') as f:
        json.dump(unique_horizontal_list, f)

    with open(f'{data_json_path}/underlines_data.json', 'w') as f:
        json.dump(unique_horizontal_list, f)
        
    with open(f'{data_csv_path}/underlines_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(unique_horizontal_list)
