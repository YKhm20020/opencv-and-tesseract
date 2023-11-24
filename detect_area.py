import os
import sys
from typing import List, Tuple
import cv2
import numpy as np
from prepare import create_area_directories, load_area_image
from export_data import export_area_data


def process_image_area(input_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """ 画像処理を行う関数
    
    グレースケール化、ガウシアンフィルタ適用、二値化、膨張処理を行う
    さらに、下線部認識のみ、エッジ検出を行う

        Args:
            input_img (numpy.ndarray): 入力画像

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]: 膨張処理後の画像、エッジ検出後の画像、二値化で決定した閾値
            
        Note:
            img_bw (numpy.ndarray): 膨張処理後の画像
            img_edges (numpy.ndarray): エッジ検出後の画像
            retval (float): 二値化で決定した閾値

    """
    img = input_img.copy()
    
    results_path = './results/rects'
    # BGR -> グレースケール
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    cv2.imwrite(f'{results_path}/0_gray.png', img_gray) # 確認用
    
    # 第四引数が cv2.THRESH_TOZERO_INV で直線をひとつ多く検出したことを確認。他サンプルと比較必須。
    #retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    retval, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f'{results_path}/1_thresh.png', img_bw) # 確認用
    
    # 膨張処理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_bw = cv2.dilate(img_bw, kernel, iterations=1)
    cv2.imwrite(f'{results_path}/2_dilate.png', img_bw) # 確認用
    
    # Canny 法によるエッジ検出（下線部検出のみ）
    results_path = './results/underlines'
    med_val = np.median(img_bw)
    sigma = 0.33
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))
    img_edges = cv2.Canny(img_bw, threshold1=min_val, threshold2=max_val, apertureSize=5, L2gradient=True)
    cv2.imwrite(f'{results_path}/3_edges.png', img_edges) # 確認用
    
    return img_bw, img_edges, retval



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


def find_rectangles(img_bw: np.ndarray, img_rects: np.ndarray, filename: str) -> np.ndarray:
    """ 矩形領域の座標を検出する関数

    矩形領域の座標を検出する関数

        Args:
            img_bw (numpy.ndarray): 膨張処理後の画像
            img_rects (numpy.ndarray): 結果出力用の画像
            filename (str): ファイルの名前
            
        Returns:
            rects_sorted_memory (numpy.ndarray): 矩形領域の座標を記録したリスト

    """
    contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
        
        color = np.random.randint(0, 255, 3).tolist()
        cv2.drawContours(img_rects, rects, i, color, 2)
        cv2.putText(img_rects, str(i), tuple(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        print(f'rect({i}):\n{rect_sorted}')
        
    print()
    
    results_path = './results/rects'
    cv2.imwrite(f'{results_path}/rects_{filename}.png', img_rects) # 結果を描画した画像の保存
    cv2.imwrite('img.png', img_rects) # 一時確認用

    rect_sorted_memory = np.array(rect_sorted_memory)
    rect_sorted_list = rect_sorted_memory.tolist()
    
    # .txt, .json, .csv ファイルで矩形領域の座標をエクスポート
    output_path = './data/rects'
    data_txt_path = os.path.join(output_path, 'txt')
    data_json_path = os.path.join(output_path, 'json')
    data_csv_path = os.path.join(output_path, 'csv')

    export_area_data(rect_sorted_list, os.path.join(data_txt_path, f'rects_data.txt'), 'txt')
    export_area_data(rect_sorted_list, os.path.join(data_json_path, f'rects_data.json'), 'json')
    export_area_data(rect_sorted_list, os.path.join(data_csv_path, f'rects_data.csv'), 'csv')

    return rect_sorted_memory



def find_underlines(img_edges: np.ndarray, img_underline: np.ndarray, rect_sorted_memory: np.ndarray, retval: float, filename: str) -> None:
    """ 
    下線部領域の座標を検出する関数
        Args:
            img_edges (numpy.ndarray): エッジ検出後の画像
            img_underline (numpy.ndarray): 結果出力用の画像
            rect_sorted_memory (numpy.ndarray): 矩形領域の座標を記録したリスト
            retval (float): 二値化で決定した閾値
            filename (str): ファイルの名前
    """
    height, width = img_edges.shape
    min_length = width * 0.1

    # ハフ変換による直線検出
    lines = []
    lines = cv2.HoughLinesP(img_edges, rho=1, theta=np.pi/360, threshold=int(retval), minLineLength=min_length, maxLineGap=1)

    line_list = []
    same_line_error = 10 # 上下に生成される直線を同一のものと捉える誤差

    if lines is None:
        print('Straight lines are not detected')
        sys.exit()
    else:
        for line in lines:
            tl_x, tl_y, br_x, br_y = line[0]
            # 傾き3px以内で検出対象に
            if abs(tl_y - br_y) < 3:
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
        rect_error = 20 # 検知した直線を矩形の一部と捉える誤差
        
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

        # 矩形領域と重複しない水平線の座標を表示する
        if unique_horizontal_nparray.shape[0] == 0:
            print('Underlines are not detected')
            sys.exit()
        else:
            for i, line in enumerate(unique_horizontal_nparray):
                x1, y1, x2, y2 = unique_horizontal_nparray[i]
                cv2.line(img_underline, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_underline, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                print(f'line({i}):\n{unique_horizontal_nparray[i]}')

        results_path = './results/underlines'
        cv2.imwrite(f'{results_path}/underline_{filename}.png', img_underline)
        cv2.imwrite('img_underline.png', img_underline) # 確認用
        
        unique_horizontal_list = unique_horizontal_nparray.tolist()
        
        # .txt, .json, .csv ファイルで下線部領域の座標をエクスポート
        output_path = './data/underlines'
        data_txt_path = os.path.join(output_path, 'txt')
        data_json_path = os.path.join(output_path, 'json')
        data_csv_path = os.path.join(output_path, 'csv')

        export_area_data(unique_horizontal_list, os.path.join(data_txt_path, f'underlines_data.txt'), 'txt')
        export_area_data(unique_horizontal_list, os.path.join(data_json_path, f'underlines_data.json'), 'json')
        export_area_data(unique_horizontal_list, os.path.join(data_csv_path, f'underlines_data.csv'), 'csv')



def main():
    # ディレクトリ作成、入力画像の決定と読み取り
    create_area_directories()
    
    try:
        #input_path =  './sample/P/3．入出退健康管理簿.pdf'
        #input_path =  './sample/P/13-3-18 入出退健康管理簿（確認印欄あり）.pdf'
        #input_path =  './sample/P/20230826_富士瓦斯資料_設備保安点検01.pdf'

        input_path = './sample/deblur_sample9.png'
        #input_path = './sample/P/02稟議書_/A281新卒者採用稟議書.png'
        #input_path = './sample/P/02稟議書_/A282広告出稿稟議書.png'
        #input_path = './sample/P/02稟議書_/A321稟議書.png'
        #input_path = './sample/P/02稟議書_/A438安全衛生推進者選任稟議書.png'
        #input_path = './sample/P/02稟議書_/A481広告出稿稟議書.png'
        #input_path = './sample/P/18作業報告書_/B090入庫報告書.png'
        #input_path = './sample/P/26休暇届_/A089夏季休暇届.png'
        
        # ファイルが存在しない場合の例外処理
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The file '{input_path}' does not exist.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit()
        
    # 入力画像の読み込み
    filename = os.path.splitext(os.path.basename(input_path))[0]
    image_original, image_rects, image_underline = load_area_image(input_path)

    # 画像処理と領域取得
    image_bw, image_edges, retval = process_image_area(image_original)
    rect_coords = find_rectangles(image_bw, image_rects, filename)
    find_underlines(image_edges, image_underline, rect_coords, retval, filename)

if __name__ == "__main__":
    main()
