import MeCab
import OCR

for i, line in enumerate(OCR.lines):
  tagger = MeCab.Tagger()
  result = tagger.parse(OCR.text_result[i])
  print(result)
  
tagger = MeCab.Tagger()
result = tagger.parse("mm")
print(result)