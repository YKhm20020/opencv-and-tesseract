from transformers import pipeline
unmasker = pipeline('fill-mask', model='line-corporation/line-distilbert-base-japanese', trust_remote_code=True)
print(unmasker("intは整数、stringは文字列、radioは単一選択、checkは複数選択のこと。氏名は、int, string, radio, check のうち、[MASK]にあてはまる。"))