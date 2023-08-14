import fasttext

model = fasttext.train_supervised(input = 'train_fasttext.txt')
model.save_model('photolize.bin')

with open('train_fasttext.txt', 'r') as f:
  text = f.read()
  words_date = text.count('date')
  words_num = text.count('num')
  words_char = text.count('char')
  words_boolean = text.count('boolean')

print('\nnumber of data:')
print(f'date: {words_date}')
print(f'num: {words_num}')
print(f'char: {words_char}')
print(f'boolean: {words_boolean}')

ret = model.predict("年齢")
print(f'{ret}\n')