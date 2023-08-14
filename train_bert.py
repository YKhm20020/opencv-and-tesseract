from transformers import BertForSequenceClassification, BertJapaneseTokenizer

# モデル
model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=9)
# model.cuda() # cudaを使う場合は、この行を有効にする
# トークナイザー
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

import os
from datasets import load_dataset

# トークナイズ用関数
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

# 学習用データ
train_data = load_dataset("csv", data_files="train_fasttext.csv", column_names=["label", "text"], split="train")
train_data = train_data.map(tokenize, batched=True, batch_size=len(train_data))
train_data.set_format("torch", columns=["input_ids", "label"])

# テスト用データ
test_data = load_dataset("csv", data_files="test_fasttext.csv", column_names=["label", "text"], split="train")
test_data = test_data.map(tokenize, batched=True, batch_size=len(test_data))
test_data.set_format("torch", columns=["input_ids", "label"]) 


# 評価用関数
from sklearn.metrics import accuracy_score

def compute_metrics(result):
    labels = result.label_ids
    preds = result.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }


# Trainerの設定
from transformers import Trainer, TrainingArguments

# 学習用パラメーター
training_args = TrainingArguments(
    output_dir = "./train_results",
    num_train_epochs = 2,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 32,
    warmup_steps = 500,  # 学習係数が0からこのステップ数で上昇
    weight_decay = 0.01,  # 重みの減衰率
    # evaluate_during_training = True,  # ここの記述はバージョンによっては必要ありません
    logging_dir = "./logs",
)

# Trainerの初期化
trainer = Trainer(
    model = model, # 学習対象のモデル
    args = training_args, # 学習用パラメーター
    compute_metrics = compute_metrics, # 評価用関数
    train_dataset = train_data, # 学習用データ
    eval_dataset = test_data, # テスト用データ
)

trainer.train() # 学習
trainer.evaluate() # 評価

model_dir = "./model"
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)




