import cv2
from IPython.display import Image, display
import numpy as np
import copy
import json

def convert_nparr_to_image(img):
    ret, encoded = cv2.imencode(".jpg", img)
    display(Image(encoded))
    
def create_pos_data(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return {'x': x, 'y': y, 'width': w, 'height': h}

#img.shapeに高さ、幅、チャンネル情報を分割代入
img = cv2.imread('sample.png')
height, width, channels = img.shape

#インプット画像をグレースケール化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Cannyを用いたエッジ検出
threshold1 = 800
threshold2 = 800
edges = cv2.Canny(img_gray, threshold1, threshold2)

#膨張処理（線を太くする）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilates = cv2.dilate(edges, kernel)

#輪郭検出
contours, hierarchy = cv2.findContours(dilates, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# エクスポート用イメージ配列を作成しておく
export_img = copy.deepcopy(img)

# JSON用の辞書型配列
export_array = {
    'width': width,
    'height': height,
    'results': []
}

for i in range(len(contours)):
    # 色を指定する
    color = np.random.randint(0, 255, 3).tolist()
    
    if cv2.contourArea(contours[i]) < 3000:
        continue  # 面積が小さいものは除く
    
    # 階層が第１じゃなかったら ... 
    if hierarchy[0][i][3] != -1:
        # 配列に追加
        export_array['results'].append(create_pos_data(contours[i]))
        # 画像に当該の枠線を追加
        cv2.drawContours(export_img, contours, i, color, 3)  
