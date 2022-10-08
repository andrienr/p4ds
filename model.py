"""
@author: enrico

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RepeatedKFold
from keras.layers import Dense
from keras.models import Sequential
from preprocess import Preprocess
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


class Model():

    """Load the saved model
    """

    def __init__(self, model_path, ui_model=True):

        self.model_path = model_path
        self.data = Preprocess()
        self.tokenizer = self.data.tokenizer
        self.categories = self.data.categories
        self.features = self.data.features
        self.df_multilabel_labels = self.data.df_multilabel_labels
        self.df_multiclass_labels = self.data.df_multiclass_labels

        if ui_model:
            # if a compiled model does not exist create a new one and train for 5 epochs
            if not os.path.exists(model_path):

                start = time.time()
                print(
                    '////////////////////////////////////////////////' +
                    'STARTING THE TRAINING, PLEASE BE PATIENT!' +
                    '////////////////////////////////////////////////'
                )

                self.train_models()

                end = time.time()
                print("Training elapsed time: ", end - start, " seconds")

            else:
                # load a already trained and compiled model from model_path directory
                self.multilabel_model = tf.keras.models.load_model(
                    os.path.join(model_path, 'multilabel_model'))
                self.multiclass_model = {}
                for cat in self.categories:
                    self.multiclass_model.update({cat: tf.keras.models.load_model(
                        os.path.join(self.model_path, 'multiclass_model', cat))})

    def get_multilabel_model(self, n_inputs, n_outputs):
        model = Sequential()
        model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(n_outputs, activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def get_multiclass_model(self, n_inputs):
        model = Sequential()
        model.add(Dense(8, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(3, activation='softmax'))
        # compile model
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False), optimizer='adam', metrics=['accuracy'])
        return model

    def train_models(self, epochs=5):
        # load dataset
        n_inputs, n_outputs = self.features.shape[1], self.df_multilabel_labels.shape[1]
        # get model
        multilabel_model = self.get_multilabel_model(n_inputs, n_outputs)
        # fit the models on all data
        multilabel_model.fit(self.features, self.df_multilabel_labels, verbose=1, epochs=epochs)
        multilabel_model.save(os.path.join(self.model_path, 'multilabel_model'))

        multiclass_model = {}
        for index, cat in enumerate(self.categories, 1):
            print('training category ' + str(cat))
            encoded_multiclass_labels = OneHotEncoder().fit_transform(
                self.df_multiclass_labels[:, index - 1].reshape(-1, 1)).toarray()
            # get model
            model = self.get_multiclass_model(n_inputs)
            mcl = pd.DataFrame(self.df_multiclass_labels)
            class_weight = {
                i: (1 / mcl.groupby(mcl[index - 1]).size().values[i]) * (len(mcl) / 3.0)
                for i in range(0, len(mcl.groupby(mcl[index - 1]).size().values))}
            # fit the models on all data
            model.fit(self.features, encoded_multiclass_labels,
                      verbose=1, epochs=epochs, class_weight=class_weight)
            model.save(os.path.join(self.model_path, 'multiclass_model', cat))
            multiclass_model.update({cat: model})

    def predict_text(self, text):
        result = {}

        # first predict the category
        multilabel_prediction = self.multilabel_model(np.asarray([self.tokenizer(text)]))
        threshold = 0.4
        relevant_categories = np.where(multilabel_prediction[0] > threshold)

        # predict the sentiment for the relevant categories
        for cat in self.categories[relevant_categories]:
            multiclass_prediction = self.multiclass_model[cat](
                np.asarray([self.tokenizer(text)]))
            if np.argmax(multiclass_prediction) == 0:
                sentiment = 'negative'
            elif np.argmax(multiclass_prediction) == 1:
                sentiment = 'neutral'
            else:
                sentiment = 'positive'
            result.update({cat: sentiment})
        return result

    def evaluate_model(self, features, labels, task, epochs, cat=None):
        n_inputs = features.shape[1]
        results = list()
        # define evaluation procedure
        cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=452326)
        pred, l_test = np.array([]), np.array([])
        for train_ids, test_ids in cv.split(features):
            # prepare data
            # define model
            if task == 'multilabel':
                features_train, features_test = features[train_ids], features[test_ids]
                labels_train, labels_test = labels[train_ids], labels[test_ids]
                n_outputs = labels.shape[1]
                model = self.get_multilabel_model(n_inputs, n_outputs)
                model.fit(features_train, labels_train, epochs=epochs)
                # make a prediction on the test set
                prediction = model.predict(features_test).round()
            elif task == 'multiclass':
                features_train, features_test = features[train_ids], features[test_ids]
                labels_train, labels_test = labels[train_ids], labels[test_ids]
                # encode class values as onehot integers
                enc = OneHotEncoder(handle_unknown='ignore')
                labels_train = enc.fit_transform(labels_train.reshape(-1, 1)).toarray()
                labels_test = enc.fit_transform(
                    labels_test.reshape(-1, 1)).toarray()
                model = self.get_multiclass_model(n_inputs)
                # use the class weights to force the model paying more attention to under-represented classes
                mcl = pd.DataFrame(self.df_multiclass_labels)
                class_weight = {
                    i: (1 / mcl.groupby(mcl[cat]).size().values[i]) * (len(mcl) / 3.0)
                    for i in range(0, len(mcl.groupby(mcl[cat]).size().values))}
                model.fit(features_train, labels_train, epochs=epochs,
                          class_weight=class_weight)
                # round probabilities to class labels
                prediction = model.predict(features_test).round()
                # print(prediction)
                self.make_confusion_matrices(
                    enc, cat, epochs, prediction, labels_test, pred, l_test)
            else:
                print('please select a correct task. \n Task supported are: multilabel, multiclass')

            # calculate accuracy
            acc = accuracy_score(labels_test, prediction)
            results.append(acc)
        return results

    def make_confusion_matrices(self, enc, cat, epochs, prediction, labels_test, pred, l_test):
        prediction = enc.inverse_transform(prediction)
        labels_test = enc.inverse_transform(labels_test)
        to_drop = np.where([prediction == None])[1]
        prediction = np.delete(prediction, to_drop, 0)
        labels_test = np.delete(labels_test, to_drop, 0)
        l_test = np.append(l_test, labels_test)
        pred = np.append(pred, prediction)
        # print(pred.shape, l_test.shape)
        disp = ConfusionMatrixDisplay.from_predictions(
            l_test,
            pred,
            cmap=plt.cm.Greens,
            normalize='true',)
        category = self.categories[cat]
        disp.ax_.set_title(
            "Normalized confusion matrix - " + str(category) + ' - trained for ' + str(epochs) +
            ' epoch(s)')
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'confusion_matrices/')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.savefig(results_dir + category)
