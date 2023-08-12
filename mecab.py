import MeCab
import OCR2

for i, line in enumerate(OCR2.lines):
  tagger = MeCab.Tagger()
  result = tagger.parse(OCR2.text_result[i])
  print(result)