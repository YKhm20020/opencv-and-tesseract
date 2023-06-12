import sys

import pyocr
import pyocr.builders

import cv2
from PIL import Image

def imageToText(src):
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)

    tool = tools[0]

    dst = tool.image_to_string(
        Image.open(src),
        lang='jpn',
        builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6)
    )
    return dst  

if __name__ == '__main__':
    img_path = sys.argv[1]

    out = imageToText(img_path)

    img = cv2.imread(img_path)
    sentence = []
    for d in out:
        sentence.append(d.content)
        cv2.rectangle(img, d.position[0], d.position[1], (0, 0, 255), 2)

    print("".join(sentence).replace("。","。\n"))

    cv2.imshow("img", img)
    cv2.imwrite("output.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
