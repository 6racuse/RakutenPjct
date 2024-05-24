from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from scipy.io import savemat

# Enregistrer la fonction f1_score
@tf.keras.utils.register_keras_serializable()
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    f1 = 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))
    return f1

class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, num_classes=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=[f1_score])
        return model

    def fit(self, X, y):
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        if self.num_classes is None:
            self.num_classes = y.shape[1]
        
        self.model = self.build_model()

        early_stopping = EarlyStopping(monitor='val_f1_score', patience=5, mode='max', restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
        model_checkpoint = ModelCheckpoint('.\models\\best_model.keras', monitor='val_f1_score', mode='max', save_best_only=True, verbose=1)

        history = self.model.fit(X, y, validation_split=0.2, epochs=4, batch_size=32,
                                 callbacks=[early_stopping, reduce_lr, model_checkpoint])

        return self

    def predict(self, X):
        y_pred_prob = self.model.predict(X)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred

    def predict_proba(self, X):
        return self.model.predict(X)

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()
    
    def fit(self, y):
        self.label_encoder.fit(y)
        return self
    
    def transform(self, y):
        y_encoded = self.label_encoder.transform(y)
        return to_categorical(y_encoded)
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)

