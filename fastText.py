import MeCab
import fasttext
import OCR2

for i, line in enumerate(OCR2.lines):
  tagger = MeCab.Tagger()
  result = tagger.parse(OCR2.text_result[i])
  print(result)

model = fasttext.train_supervised(input="data.txt")
model.save_model("photolize.bin")

ret = model.predict(MeCab.result)
print(ret)