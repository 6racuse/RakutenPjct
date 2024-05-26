import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

#SVM
def train_model(X_train_tfidf, Y_train):
    param_grid = {
        'C': 8.071428571428571,
        'gamma': 0.1,
        'kernel': 'rbf'
    }
    svm = SVC(
        C=param_grid['C'],
        gamma=param_grid['gamma'],
        kernel=param_grid['kernel']
    )
    svm.fit(
        X_train_tfidf,
        Y_train
    )
    return svm

#kNN
def train_knn(X_train, y_train):
    BestKNN = KNeighborsClassifier(n_neighbors=49)
    BestKNN.fit(X_train, y_train)

    return BestKNN

#NN
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