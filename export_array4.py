import cv2
import numpy as np

img = cv2.imread('./sample3.png')

# BGR -> グレースケール
#img = cv2.blur(img, (3,3))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.blur(img_gray, (3,3))
img_bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 2101, 10)
cv2.imwrite('result1.png', img_bw)

# 膨張処理
kernel = np.ones((2,2), np.uint8)
img_bw = cv2.dilate(img_bw, kernel, iterations = 1)
cv2.imwrite('result2.png', img_bw)

# 輪郭抽出
contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 面積でフィルタリング
rects = []
for cnt, hrchy in zip(contours, hierarchy[0]):
    if cv2.contourArea(cnt) < 1500:
        continue  # 面積が小さいものは除く
    if hrchy[3] == -1:
        continue  # ルートノードは除く
   
    # 輪郭を囲む長方形を計算する。
    rect = cv2.minAreaRect(cnt)
    rect_points = cv2.boxPoints(rect).astype(int)
    rects.append(rect_points)

# x-y 順でソート
rects = sorted(rects, key=lambda x: (x[0][1], x[0][0]))

# 描画する。
for i, rect in enumerate(rects):
    color = np.random.randint(0, 255, 3).tolist()
    cv2.drawContours(img, rects, i, color, 2)
    cv2.putText(img, str(i), tuple(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
    
    print('rect(%d):\n' %i, rect)

cv2.imwrite('img.png', img)