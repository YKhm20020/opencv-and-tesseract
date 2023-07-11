import cv2
import numpy as np

# 入力画像の決定と読み取り
inputImage = './sample/sample.jpg'

img = cv2.imread(inputImage)

# BGR -> グレースケール
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imwrite('result1.png', img_bw)

# 膨張処理
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
img_bw = cv2.dilate(img_bw, kernel, iterations = 1)
# cv2.imwrite('result2.png', img_bw)

# 2 値化
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
thresh = cv2.bitwise_not(thresh)

# 輪郭抽出（外接矩形取得）
exContours, exHierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in exContours:
    x, y, xx, yy = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x,y), (x+xx, y+yy), (0,0,255),10)

cv2.imwrite("cropped_edge_rectangle.jpg", img)

# 輪郭抽出
contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# i = 1 は画像全体の外枠になるのでカウントに入れない
x1 = []
y1 = []
x2 = []
y2 = []
for i in range(1, len(exContours)):
    # ret の中身は (x, y, w, h)
    ret = cv2.boundingRect(exContours[i])
    x1.append(ret[0])
    y1.append(ret[1])
    x2.append(ret[0] + ret[2])
    y2.append(ret[1] + ret[3])

x1_min = min(x1)
y1_min = min(y1)
x2_max = max(x2)
y2_max = max(y2)

# 枠取りをした結果を表示
cv2.rectangle(img, (x1_min, y1_min), (x2_max, y2_max), (0, 255, 0), 2)
cv2.imwrite('cropped_edge_rectangle.jpg', img)

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
    
"""" 以下線認識機能

lines = cv2.HoughLinesP(img_bw, rho=1, theta=np.pi/360, threshold=80, minLineLength=400, maxLineGap=5)
print(lines)

for line in lines:
    x1, y1, x2, y2 = line[0]
    
    line_img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
cv2.imwrite("img_lines.png", img)

"""

# x-y 順でソート
rects = sorted(rects, key=lambda x: (x[0][1], x[0][0]))

# 描画する。
for i, rect in enumerate(rects):
    color = np.random.randint(0, 255, 3).tolist()
    cv2.drawContours(img, rects, i, color, 2)
    cv2.putText(img, str(i), tuple(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    print('rect(%d):\n' %i, rect)

cv2.imwrite('img.png', img)