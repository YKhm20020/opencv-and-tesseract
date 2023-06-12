from PIL import Image
import pyocr
import pyocr.builders
 
# OCRエンジンの取得
tools = pyocr.get_available_tools()
tool = tools[0]
 
# 画像の読み込み
img_org = Image.open("./sample/sample.png")
 
# OCRの実行
builder = pyocr.builders.TextBuilder()
result = tool.image_to_string(img_org, lang="jpn", builder=builder)
 
print(result)