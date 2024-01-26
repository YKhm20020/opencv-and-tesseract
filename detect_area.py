import os
import sys
from typing import List, Tuple
import cv2
import numpy as np
from prepare import create_area_directories, load_area_image
from export_data import export_rects_data, export_underlines_data


def process_image_rect(input_img: np.ndarray) -> Tuple[np.ndarray, float]:
    """ 画像処理を行う関数
    
    グレースケール化、ガウシアンフィルタ適用、二値化、膨張処理を行う
    さらに、下線部認識のみ、エッジ検出を行う

        Args:
            input_img (numpy.ndarray): 入力画像

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]: 膨張処理後の画像、エッジ検出後の画像、二値化で決定した閾値
            
        Note:
            img_bw (numpy.ndarray): 膨張処理後の画像
            retval (float): 二値化で決定した閾値

    """
    
    img = input_img.copy()
    results_path = './results/rects'

    # BGR -> グレースケール
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    cv2.imwrite(f'{results_path}/0_gray.png', img_gray) # 確認用
    
    # 二値化処理
    retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f'{results_path}/1_thresh.png', img_bw) # 確認用
    
    # 膨張処理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_bw = cv2.dilate(img_bw, kernel, iterations=1)
    cv2.imwrite(f'{results_path}/2_dilate.png', img_bw) # 確認用
    
    return img_bw, retval

def process_image_underline(input_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """ 画像処理を行う関数
    
    グレースケール化、二値化、膨張処理、エッジ検出を行う

        Args:
            input_img (numpy.ndarray): 入力画像

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]: 膨張処理後の画像、エッジ検出後の画像、二値化で決定した閾値
            
        Note:
            img_bw (numpy.ndarray): 二値化処理後の画像
            img_edges (numpy.ndarray): エッジ検出後の画像
            retval (float): 二値化で決定した閾値

    """

    img = input_img.copy()
    results_path = './results/underlines'

    # BGR -> グレースケール
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{results_path}/0_gray.png', img_gray) # 確認用
    
    # 二値化処理
    retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    cv2.imwrite(f'{results_path}/1_thresh.png', img_bw) # 確認用
    
    img_bw_inv = cv2.bitwise_not(img_bw)
    
    # Canny 法によるエッジ検出（下線部検出のみ）
    # med_val = np.median(img_bw_inv)
    # sigma = 0.33
    # min_val = int(max(0, (1.0 - sigma) * med_val))
    # max_val = int(max(255, (1.0 + sigma) * med_val))
    # img_edges = cv2.Canny(img_bw, threshold1=min_val, threshold2=max_val, apertureSize=5, L2gradient=True)
    # cv2.imwrite(f'{results_path}/3_edges.png', img_edges) # 確認用
    
    return img_bw_inv, retval


def sort_points(points: np.ndarray) -> List[np.ndarray]:
    """ 頂点を並び替える関数
    
    頂点を左上、左下、右下、右上の順序に並び替える関数
    
        Args:
            points (numpy.ndarray): 矩形領域の座標を格納したリスト
            
        Returns:
            list[numpy.ndarray]: 並び替え後の矩形の頂点座標
            
        Note: 
            tl : top left (左上) の点の x, y 座標
            tr : top right (右上) の点の x, y 座標
            br : bottom right (右下) の点の x, y 座標
            bl : bottom left (左下) の点の x, y 座標

    """
    
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


def find_rectangles(img_bw: np.ndarray, img_rects: np.ndarray, file_name: str) -> np.ndarray:
    """ 矩形領域の座標を検出する関数

    矩形領域の各頂点の x, y 座標を検出する関数

        Args:
            img_bw (numpy.ndarray): 膨張処理後の画像
            img_rects (numpy.ndarray): 結果出力用の画像
            file_name (str): ファイルの名前
            
        Returns:
            rects_sorted_memory (numpy.ndarray): 矩形領域の座標を記録したリスト

    """
    
    contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        # 面積でフィルタリング
        rects = []
        for cnt, hrchy in zip(contours, hierarchy[0]): 
            if cv2.contourArea(cnt) < 3000:
                continue  # 面積が小さいものを除外
            if hrchy[3] == -1:
                continue  # ルートノードを除外
        
            # 輪郭を囲む長方形を計算
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

        # 矩形領域を描画する
        for i, rect in enumerate(rects):
            # 頂点を左上、左下、右下、右上の順序に並び替える
            rect_sorted = np.array(sort_points(rect))
            
            rect_sorted_memory.append(rect_sorted)
            
            # color = np.random.randint(0, 255, 3).tolist()
            # cv2.drawContours(img_rects, rects, i, color, 2)
            # cv2.putText(img_rects, str(i), tuple(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # print(f'rect({i}):\n{rect_sorted}')
        
        # results_path = './results/rects'
        # cv2.imwrite(f'{results_path}/rects_{file_name}.png', img_rects) # 結果を描画した画像の保存
        # cv2.imwrite('img.png', img_rects) # 一時確認用

        rect_sorted_memory = np.array(rect_sorted_memory)
        rect_sorted_list = rect_sorted_memory.tolist()
        
        # 矩形領域の座標をファイルにエクスポート
        export_rects_data(rect_sorted_list, file_name)
        
        return rect_sorted_memory


def find_underlines(img_bw_inv: np.ndarray, img_underline: np.ndarray, rect_sorted_memory: np.ndarray, retval: float, file_name: str) -> List[np.ndarray]:
    """ 下線部領域の座標を検出する関数
    
    下線の両端点の x, y 座標を出力する関数
    
        Args:
            img_bw_inv (numpy.ndarray): 白黒反転した画像
            img_underline (numpy.ndarray): 結果出力用の画像
            rect_sorted_memory (numpy.ndarray): 矩形領域の座標を記録したリスト
            retval (float): 二値化で決定した閾値
            file_name (str): ファイルの名前
            
        Returns:
            List[numpy.ndarray]: 下線の両端点の座標
    """
    
    height, width = img_bw_inv.shape
    min_length = width * 0.1
    
    length_threshold = 60 # 30 ～ 100
    distance_threshold = 1.41421356
    
    med_val = retval
    sigma = 0.33  # 0.33
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))

    canny_th1 = min_val
    canny_th2 = max_val
    canny_aperture_size = 3
    do_merge = True
    
    # print(med_val, min_val, max_val, canny_th1, canny_th2)

    # ハフ変換による直線検出
    lines = []
    # lines = cv2.HoughLinesP(img_edges, rho=1, theta=np.pi/360, threshold=int(retval), minLineLength=min_length, maxLineGap=1)
    fld = cv2.ximgproc.createFastLineDetector(
        length_threshold,
        distance_threshold,
        canny_th1,
        canny_th2,
        canny_aperture_size,
        do_merge
    )
    lines = fld.detect(img_bw_inv)

    line_list = []
    same_line_error = 10 # 上下に生成される直線を同一のものと捉える誤差

    if lines is None:
        print('Straight lines are not detected')
        return None
    else:
        for line in lines:
            left_x, left_y, right_x, right_y = line[0]
            
            # createFastLineDetector 関数のみ端点入れ替わるときがあるので、左→右になるよう入れ替え
            if left_x > right_x:
                tmp_x, tmp_y = left_x, left_y
                left_x, left_y = right_x, right_y
                right_x, right_y = tmp_x, tmp_y
                # left_x, left_y = left_y, left_x
                # right_x, right_y = right_y, right_x

            # 傾き3px以内で検出対象に
            if abs(left_y - right_y) < 3:
                line_list.append((left_x, left_y, right_x, right_y))
                
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

        # # 矩形領域と重複しない水平線の座標を表示する
        # if unique_horizontal_nparray.shape[0] == 0:
        #     print('Underlines are not detected')
        #     return None
        # else:
        #     for i in range(len(unique_horizontal_nparray)):
        #         x1, y1, x2, y2 = unique_horizontal_nparray[i]
        #         cv2.line(img_underline, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #         cv2.putText(img_underline, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        #         print(f'line({i}):\n{unique_horizontal_nparray[i]}')

        # results_path = './results/underlines'
        # cv2.imwrite(f'{results_path}/underline_{file_name}.png', img_underline)
        # cv2.imwrite('img_underline.png', img_underline) # 確認用
        
        unique_horizontal_list = unique_horizontal_nparray.tolist()
        
        # 下線部領域の座標をファイルにエクスポート
        export_underlines_data(unique_horizontal_list, file_name)
        
        return unique_horizontal_list



def main():
    # ディレクトリ作成
    create_area_directories()
    
    try:
        #input_path =  './sample/P/3．入出退健康管理簿.pdf'
        #input_path =  './sample/P/13-3-18 入出退健康管理簿（確認印欄あり）.pdf'
        #input_path =  './sample/P/20230826_富士瓦斯資料_設備保安点検01.pdf'

        #input_path = './sample/sample2.jpg'
        input_path = './sample/seikyuu.jpg'
        #input_path = './sample/sample.png'
        #input_path = './sample/P/（158-000306）自動車保険契約内容変更依頼書/作成/【ベース】AA300319_1-1.jpg'
        #input_path = './sample/P/（158-000306）自動車保険契約内容変更依頼書/作成/変更_AA300319.pdf'
        #input_path = './sample/P/02稟議書_/A281新卒者採用稟議書.png'
        #input_path = './sample/P/02稟議書_/A282広告出稿稟議書.png'
        #input_path = './sample/P/02稟議書_/A321稟議書.png'
        #input_path = './sample/P/02稟議書_/A438安全衛生推進者選任稟議書.png'
        #input_path = './sample/P/02稟議書_/A481広告出稿稟議書.png'
        #input_path = './sample/P/18作業報告書_/B090入庫報告書.png'
        #input_path = './sample/P/26休暇届_/A089夏季休暇届.png'
        
        # ファイルが存在しない場合の例外処理
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"The file '{input_path}' does not exist.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit()
        
    # 入力画像の読み込み
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    image_original, image_rects, image_underline = load_area_image(input_path)

    # 画像処理と領域取得
    image_bw, retval = process_image_rect(image_original)
    image_bw_inv, retval = process_image_underline(image_original)
    
    rect_coords = find_rectangles(image_bw, image_rects, file_name)
    
    if rect_coords is None:
        print("No contours found")
    else:
        # 矩形領域を描画する
        for i, rect in enumerate(rect_coords):
            # 頂点を左上、左下、右下、右上の順序に並び替える
            rect_sorted = np.array(sort_points(rect))        
            color = np.random.randint(0, 255, 3).tolist()
            cv2.drawContours(image_rects, rect_coords, i, color, 2)
            #cv2.drawContours(image_rects, rect_coords, i, (255,255,255), 12)
            cv2.putText(image_rects, str(i), tuple(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            print(f'rect({i}):\n{rect_sorted}')
        
        print()
    
        results_path = './results/rects'
        cv2.imwrite(f'{results_path}/rects_{file_name}.png', image_rects) # 結果を描画した画像の保存
        cv2.imwrite('img.png', image_rects) # 一時確認用
    
    underline_coords = find_underlines(image_bw_inv, image_underline, rect_coords, retval, file_name)
    
    if underline_coords is None:
        print('Underlines are not detected')
    else:
        for i in range(len(underline_coords)):
            x1, y1, x2, y2 = underline_coords[i]
            cv2.line(image_underline, (x1, y1), (x2, y2), (255, 255, 255), 12)
            #cv2.line(image_underline, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_underline, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            print(f'line({i}):\n{underline_coords[i]}')
            
        results_path = './results/underlines'
        cv2.imwrite(f'{results_path}/underline_{file_name}.png', image_underline)
        cv2.imwrite('img_underline.png', image_underline) # 確認用

if __name__ == "__main__":
    main()
