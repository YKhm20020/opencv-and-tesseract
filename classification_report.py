# モデル評価
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras_bert import get_custom_objects
from IPython.core.display import display
import numpy as np
import pandas as pd
import train_keras_bert as tkb

model = load_model(tkb.model_filename, custom_objects=get_custom_objects())

predicted_test_labels = model.predict([tkb.data['test_features'], tkb.data['test_segments']]).argmax(axis=1)
numeric_test_labels = np.array(tkb.data['test_labels']).argmax(axis=1)

report = classification_report(
        numeric_test_labels, predicted_test_labels, target_names=['date', 'num', 'string', 'radio', 'check'], output_dict=True)

display(pd.DataFrame(report).T)