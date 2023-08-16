import sys
import pandas as pd
import sentencepiece as spm
import logging
import numpy as np

from keras import utils
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import get_custom_objects
from sklearn.metrics import classification_report, confusion_matrix


sys.path.append('modules')

# SentencePieceProccerモデルの読込
spp = spm.SentencePieceProcessor()
spp.Load('./bert-wiki-ja/wiki-ja.model')
# BERTの学習したモデルの読込
model_filename = './bert/models/knbc_finetuning.model'
model = load_model(model_filename, custom_objects=get_custom_objects())

SEQ_LEN = 16
maxlen = SEQ_LEN

def _get_indice(feature):
    indices = np.zeros((maxlen), dtype=np.int32)

    tokens = []
    tokens.append('[CLS]')
    tokens.extend(spp.encode_as_pieces(feature))
    tokens.append('[SEP]')

    for t, token in enumerate(tokens):
        if t >= maxlen:
            break
        try:
            indices[t] = spp.piece_to_id(token)
        except:
            logging.warn('unknown')
            indices[t] = spp.piece_to_id('<unk>')
    return indices

feature = "氏名"

test_features = []
test_features.append(_get_indice(feature))
test_segments = np.zeros(
    (len(test_features), maxlen), dtype=np.float32)

# predicted_test_labels = model.predict(
#     [test_features, test_segments]).argmax(axis=1) # エラー箇所

# predicted_test_labels = model.predict(test_features).argmax(axis=1) # xだけ入力する
# test_labels = np.argmax(test_segments, axis=1) # yを別に作る

predicted_test_labels = model.predict([np.array(test_features), test_segments]).argmax(axis=1) # xとyをリストで入力する

label_data = pd.read_csv('./bert/data/id_to_labels.csv')
label = label_data.query(f'id == {predicted_test_labels[0]}')
label = label.iloc[0]
label_name = label['label']

print(f'予測対象：{feature}')
print(f'予測結果：{label_name}')