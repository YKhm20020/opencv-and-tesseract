import cv2
import numpy as np

img = cv2.imread('./sample7.jpg')


# BGR -> グレースケール
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# エッジ抽出 (Canny)
edges = cv2.Canny(gray, 100, 400, apertureSize=3)
cv2.imwrite('edges.png', edges)

# 膨張処理
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges = cv2.dilate(edges, kernel)

# 輪郭抽出
contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# 面積でフィルタリング
rects = []
for i, (cnt, hrchy) in enumerate(zip(contours, hierarchy[0])):
    if cv2.contourArea(cnt) < 3000:
        continue  # 面積が小さいものを除外
    if hrchy[0] == -1 and hrchy[1] == -1:
        continue  # 余計に検出される外側の矩形領域を除外
    if hrchy[3] == -1:
        continue  # ルートノードを除外

    # 輪郭を囲む長方形を計算する。
    rect = cv2.minAreaRect(cnt)
    rect_points = cv2.boxPoints(rect).astype(int)
    rects.append(rect_points)

# x-y 順でソート
rects = sorted(rects, key=lambda x: (x[0][1], x[0][0]))

# 描画する。
for i, rect in enumerate(rects):
    color = np.random.randint(0, 255, 3).tolist()
    cv2.drawContours(img, [rect], -1, color, 2)
    cv2.putText(img, str(i), tuple(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    print('rect(%d):\n' % i, rect)

cv2.imwrite('img.png', img)
