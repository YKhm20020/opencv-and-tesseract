import pandas as pd
import csv
import MeCab

from sklearn.model_selection import train_test_split
train_data = pd.read_table('train_data.csv', header=None,quoting=csv.QUOTE_NONE)
train, test = train_test_split(train_data, test_size=0.2, random_state=0)

train.to_csv('train.tsv', sep='\t', index=False, header=None)
test.to_csv('test.tsv', sep='\t', index=False, header=None)

# 学習データを加工
with open('train.tsv', 'r') as f_in, open('train_fasttext.txt', 'w') as f_out:
    for row in f_in:
        label, text = row.strip().split(',')
        mecab = MeCab.Tagger("-Owakati")
        text = mecab.parse(text)
        f_out.write('__label__{} {}'.format(label, text))
        
# テストデータを加工
with open('test.tsv', 'r') as f_in, open('test_fasttext.txt', 'w') as f_out:
    for row in f_in:
        label, text = row.strip().split(',')
        mecab = MeCab.Tagger("-Owakati")
        text = mecab.parse(text)
        f_out.write('__label__{} {}'.format(label, text))