from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras import backend as K
from numpy import ceil, argmax
from ZGlobal_parameter import Error_Map
from ZManageData import clean_console

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

def LaunchNN_Model(X_data, y_data):
    Bypass_bool = input("Bypass learning process - this directly loads the best_model_nn from memory - (yes/no) : ")
    if Bypass_bool == 'no':
        model = NN(X_data, y_data)
    elif Bypass_bool == 'yes':
        model = Bypass_learning()
    else:
        print("wrong input")
        return Error_Map.TYPE_ERROR_INPUT.value

    model.load_weights('best_model.keras')
    return model

def Bypass_learning():
    model = tf.keras.models.load_model('best_model.keras', custom_objects={"f1_score": f1_score})
    return model

def Get_NN_Prediction(model, X_test):
    model.load_weights('best_model.keras')

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf_test = tfidf_vectorizer.fit_transform(X_test)

    class SparseMatrixPredictionGenerator(Sequence):
        def __init__(self, X, batch_size):
            self.X = X
            self.batch_size = batch_size

        def __len__(self):
            return int(ceil(self.X.shape[0] / float(self.batch_size)))

        def __getitem__(self, idx):
            batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x.toarray()

    batch_size = 32
    test_generator = SparseMatrixPredictionGenerator(X_tfidf_test, batch_size)
    y_pred_prob = model.predict(test_generator)
    y_pred = argmax(y_pred_prob, axis=1)

    return y_pred

def NN(X_data, y_data):
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(X_data)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)

    y_categorical = to_categorical(y_encoded)

    X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y_categorical, test_size=0.2, random_state=42)

    class SparseMatrixBatchGenerator(Sequence):
        def __init__(self, X, y, batch_size):
            self.X = X
            self.y = y
            self.batch_size = batch_size

        def __len__(self):
            return int(ceil(self.X.shape[0] / float(self.batch_size)))

        def __getitem__(self, idx):
            batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x.toarray(), batch_y

    input_dim = X_train.shape[1]
    num_classes = y_categorical.shape[1]

    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[f1_score])

    early_stopping = EarlyStopping(monitor='val_f1_score', patience=5, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_f1_score', mode='max', save_best_only=True, verbose=1)

    batch_size = 32
    train_generator = SparseMatrixBatchGenerator(X_train, y_train, batch_size)
    val_generator = SparseMatrixBatchGenerator(X_val, y_val, batch_size)
    clean_console()
    history = model.fit(train_generator,
                        epochs=3,
                        validation_data=val_generator,
                        callbacks=[early_stopping, reduce_lr, model_checkpoint])

    model.save('best_model.keras')
    return model
