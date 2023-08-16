import csv

adding_row = ['label', 'feature']

# 元のファイルを読み込む
with open('./train_data.csv', 'r') as f:
    reader = csv.reader(f)
    # データをリストに変換
    data = list(reader)

# 追記したいデータを先頭に挿入
data.insert(0, adding_row)

# 新しいファイルに書き込む
with open('./bert/data/train_bert_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)

# label列用の別ファイル labels.csv を書き込みモードで開く
with open('./bert/data/labels.csv', 'w') as f: 
  writer = csv.writer(f)
  # label列を抽出して書き込む
  writer.writerows([[row[0]] for row in data])

# feature列用の別ファイル features.csv を書き込みモードで開く
with open('./bert/data/features.csv', 'w') as f: 
  writer = csv.writer(f)
  # feature列を抽出して書き込む
  writer.writerows([[row[1]] for row in data])