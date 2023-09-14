from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import numpy as np
import cv2
from loguru import logger

def japanese_puttext(img, text, position, font, fill = (255, 0, 0)):
    """ cv2.putTextが日本語対応してないので、自分で関数を定義する"""
    img_pil = Image.fromarray(img) # PIL Imageに変換。
    draw = ImageDraw.Draw(img_pil) # drawインスタンスを生成
    draw.text(position, text, font = font , fill = fill) # drawにテキストをのせる
    img = np.array(img_pil) # PILを配列に変換
    return img

def run_ocr(img_path):
    """ OCRメイン関数"""
    #画像読み込み＋前処理(適当)+PaddleOCR入力用にnpへ
    img_path = './sample/sample.jpg'
    im = Image.open(img_path).convert('L')
    enhancer= ImageEnhance.Contrast(im) #コントラストを上げる
    im_con = enhancer.enhance(2.0) #コントラストを上げる
    np_img = np.asarray(im_con)
    logger.debug('画像読み込み完了') #logは今までと変わらない！

    #PaddleOCRを定義
    ocr = PaddleOCR(
        use_gpu=False, #GPUあるならTrue
        lang = "japan", #英語OCRならen
        det_limit_side_len=im_con.size[1], #画像サイズが960に圧縮されないように必須設定
        max_text_length = 30, #検証してないがテキスト最大長のパラメータ。今回は不要だが紹介
        )
    logger.debug('PaddleOCR設定完了')

    #PaddleOCRでOCR ※cls(傾き設定)は矩形全体での補正なので1文字1文字の補正ではない為不要
    result = ocr.ocr(img = np_img, det=True, rec=True, cls=False)
    logger.debug('PaddleOCR実行完了')

    #OCR結果転記用
    result_img = np_img.copy()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    #画像に載せる日本語フォント設定　★Windows10だと C:\Windows\Fonts\以下にフォントがありまｓ
    #fontpath ='C:\Windows\Fonts\HGRPP1.TTC'
    fontpath = './HGRSGU.TTC'
    font = ImageFont.truetype(fontpath, 10) #サイズ指定

    #OCR結果を画像に転記
    for detection in result[0]:
        t_left = tuple([int(i) for i in detection[0][0]]) #左上
        # t_right = tuple([int(i) for i in detection[0][1]]) #右上
        b_right = tuple([int(i) for i in detection[0][2]]) #右下
        b_left = tuple([int(i) for i in detection[0][3]]) #左下
        ocr_text = detection[1][0] #テキスト(detection[1][1]なら自信度取得も可能)
        #画像に文字範囲の矩形を載せる(緑色)
        result_img = cv2.rectangle(result_img, t_left, b_right, (0, 255, 0), 3)
        """putTextだと日本語が??になってしまうので自作関数で処理。文字の位置は左下とした"""
        # result_img = cv2.putText(result_img, ocr_text, t_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        result_img = japanese_puttext(result_img, ocr_text, (b_left[0], b_left[1]), font)

    logger.debug('画像にOCR結果記載完了')
    #保存する
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./out/ocr_result_picture.png', result_img)
    logger.debug('結果画像の保存完了')

if __name__ == '__main__':
    # loggerを定義
    logger.add("./out/samplelog.log", format="[{time:HH.mm:ss}] <lvl>{message}</lvl>", level='DEBUG', enqueue=True)
    img_path = "./data/miyazawa.png"
    run_ocr(img_path)
    # loggerを終了
    logger.remove()