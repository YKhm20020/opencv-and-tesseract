import numpy as np
import pandas as pd
import sentencepiece as spm
import logging
import keras
from keras_bert import load_trained_model_from_checkpoint
from keras.utils import to_categorical

# BERTのロード
config_path = './bert-wiki-ja/bert_finetuning_config_v1.json'
# 拡張子まで記載しない
checkpoint_path = './bert-wiki-ja/model.ckpt-1400000'

# 最大のトークン数
SEQ_LEN = 16
BATCH_SIZE = 16
BERT_DIM = 768
LR = 1e-4
# 学習回数
EPOCH = 20

bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True,  trainable=True, seq_len=SEQ_LEN)
bert.summary()


sp = spm.SentencePieceProcessor()
sp.Load('./bert-wiki-ja/wiki-ja.model')

maxlen = SEQ_LEN

def _get_indice(feature):
    indices = np.zeros((maxlen), dtype = np.int32)

    tokens = []
    tokens.append('[CLS]')
    tokens.extend(sp.encode_as_pieces(feature))
    tokens.append('[SEP]')

    for t, token in enumerate(tokens):
        if t >= maxlen:
            break
        try:
            indices[t] = sp.piece_to_id(token)
        except:
            logging.warn(f'{token} is unknown.')
            indices[t] = sp.piece_to_id('<unk>')

    return indices

def _load_labeldata(train_dir, test_dir):
    train_features_df = pd.read_csv(f'{train_dir}/features_train.csv')
    train_labels_df = pd.read_csv(f'{train_dir}/labels_train.csv')
    test_features_df = pd.read_csv(f'{test_dir}/features_test.csv')
    test_labels_df = pd.read_csv(f'{test_dir}/labels_test.csv')
    label2index = {k: i for i, k in enumerate(train_labels_df['label'].unique())}
    index2label = {i: k for i, k in enumerate(train_labels_df['label'].unique())}
    class_count = len(label2index)
    train_labels = keras.utils.to_categorical([label2index[label] for label in train_labels_df['label']], num_classes=class_count)
    test_label_indices = [label2index[label] for label in test_labels_df['label']]
    test_labels = keras.utils.to_categorical(test_label_indices, num_classes=class_count)

    train_features = []
    test_features = []

    for feature in train_features_df['feature']:
        train_features.append(_get_indice(feature))
    train_segments = np.zeros((len(train_features), maxlen), dtype = np.float32)
    for feature in test_features_df['feature']:
        test_features.append(_get_indice(feature))
    test_segments = np.zeros((len(test_features), maxlen), dtype = np.float32)

    print(f'Trainデータ数: {len(train_features_df)}, Testデータ数: {len(test_features_df)}, ラベル数: {class_count}')
    print(f'ラベル種類: {train_labels_df["label"].unique()}')

    return {
        'class_count': class_count,
        'label2index': label2index,
        'index2label': index2label,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'test_label_indices': test_label_indices,
        'train_features': np.array(train_features),
        'train_segments': np.array(train_segments),
        'test_features': np.array(test_features),
        'test_segments': np.array(test_segments),
        'input_len': maxlen
    }
    

from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, GlobalMaxPooling1D
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras import Input, Model
from keras_bert import calc_train_steps
import tensorflow as tf


def _create_model(input_shape, class_count):
    decay_steps, warmup_steps = calc_train_steps(
        input_shape[0],
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
    )

    bert_last = bert.get_layer(name='NSP-Dense').output
    x1 = bert_last
    output_tensor = Dense(class_count, activation='softmax')(x1)
    # Trainableの場合は、Input Masked Layerが3番目の入力なりますが、
    # FineTuning時には必要無いので1, 2番目の入力だけ使用します。
    # Trainableでなければkeras-bertのModel.inputそのままで問題ありません。
    model = Model([bert.input[0], bert.input[1]], output_tensor)
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                  #optimizer='nadam',
                  metrics=['mae', 'mse', 'acc'])

    return model


# データロードとモデルの準備
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

trains_dir = 'bert/data/trains'
tests_dir = 'bert/data/tests'

data = _load_labeldata(trains_dir, tests_dir)
model_filename = 'bert/models/knbc_finetuning.model'
model = _create_model(data['train_features'].shape, data['class_count'])

model.summary()

history = model.fit([data['train_features'], data['train_segments']],
          data['train_labels'],
          epochs = EPOCH,
          batch_size = BATCH_SIZE,
          validation_data=([data['test_features'], data['test_segments']], data['test_labels']),
          shuffle=False,
          verbose = 1,
          callbacks = [
              ModelCheckpoint(monitor='val_acc', mode='max', filepath=model_filename, save_best_only=True)
          ])

print('success')