# 最大トークンの数を調べる。
# max_position_embeddings, max_seq_length の値にかかわる。

import pandas as pd
import sentencepiece as spm

# feature.csvは上記で用意したファイルのパスを指定してください
train_features_df = pd.read_csv('./bert/data/features.csv')

def _get_feature_indice(feature):
    tokens = []
    tokens.append('[CLS]')
    tokens.extend(sp.encode_as_pieces(feature))
    tokens.append('[SEP]')
    number = len(tokens)

    return number

def _get_label_indice(label):
    tokens = []
    tokens.append('[CLS]')
    tokens.extend(sp.encode_as_pieces(label))
    tokens.append('[SEP]')
    number = len(tokens)

    return number

sp = spm.SentencePieceProcessor()
# ダウンロードした事前学習モデルのパスを指定
sp.Load('./bert-wiki-ja/wiki-ja.model')

feature_numbers = []
label_numbers = []

for feature in train_features_df['feature']:
    features_number = _get_feature_indice(feature)
    feature_numbers.append(features_number)
    
train_labels_df = pd.read_csv('./bert/data/labels.csv')
    
for label in train_labels_df['label']:
    labels_number = _get_label_indice(label)
    label_numbers.append(labels_number)

# 最大トークン数
max_feature_token_num = max(feature_numbers)
max_label_token_num = max(label_numbers)

print("max_feature_token_number: " + str(max_feature_token_num))
print("max_label_token_number: " + str(max_label_token_num))