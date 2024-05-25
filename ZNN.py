import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical



def predict_labels(NN_model, X_test_tfidf,label_encoder):
    y_test_pred_proba = NN_model.predict(X_test_tfidf)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)

    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

    return y_test_pred_labels



def f1_m(y_true, y_pred):
    y_true = tf.cast(tf.argmax(y_true, axis=1), tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.float32)

    tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())

    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return tf.reduce_mean(f1)

def train__(X_train_tfidf, y_train_encoded):

    num_classes = len(np.unique(y_train_encoded))
    y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)

    model = Sequential()
    model.add(Dense(512, input_dim=X_train_tfidf.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m])

    checkpoint = ModelCheckpoint('./models/nn_model.keras', monitor='val_f1_m', mode='max', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_f1_m', mode='max', patience=5, verbose=1)

    model.fit(X_train_tfidf, y_train_categorical, epochs=3, batch_size=32, validation_split=0.2,
              callbacks=[checkpoint, early_stopping])

    best_model = tf.keras.models.load_model('./models/nn_model.keras', custom_objects={'f1_m': f1_m})

    return best_model