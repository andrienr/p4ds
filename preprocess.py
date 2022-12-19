"""
@author: enrico

"""
import time
import pandas as pd
import numpy as np
import tensorflow as tf


class Preprocess():

    """Read data from file and create an object containing preprocessed data.
    """

    def __init__(self):

        start = time.time()

        VOCAB_SIZE = 1000
        MAX_NUM_WORDS = 200

        # read the csv file
        csv = pd.read_csv('data/absita_2018_training.csv', sep=';')

        self.tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE, output_sequence_length=MAX_NUM_WORDS)
        self.tokenizer.adapt(csv.sentence.map(lambda text: text))

        self.features = self.tokenizer(csv.sentence).numpy()

        # create multilabel classification dataframe
        df_multilabel = pd.DataFrame(csv.iloc[:, -1])
        df = pd.DataFrame(csv.iloc[:, -1])
        for i in csv.columns[0:-1]:
            if 'presence' in i:
                df_multilabel = pd.concat([df_multilabel, csv[i]], axis=1)
            else:
                df = pd.concat([df, csv[i]], axis=1)
        # write multilabel classification dataframe to file
        df_multilabel.to_csv('data/df_multilabel.csv', index=False)

        # create multiclass classification dataframe
        df_multiclass = pd.DataFrame(df.iloc[:, 0])

        # if a review is both positive and negative then is considered neutral
        for i in range(2, df.columns.size, 2):
            conditions = [(df.iloc[:, i + 1] > 0),
                          ((df.iloc[:, i + 1] == 0) &
                           (df.iloc[:, i] == 0)),
                          ((df.iloc[:, i + 1] == 1) &
                           (df.iloc[:, i] == 1)),
                          (df.iloc[:, i] > 0), ]

            values = ['negative', 'neutral', 'neutral', 'positive']
            df_multiclass[df.columns[i].rsplit("_")[0]] = np.select(conditions, values)

        # write binary classification dataframe to file
        df_multiclass.to_csv('data/df_multiclass.csv', index=False)

        self.categories = df_multiclass.columns[1:]

        # transform to numpy array
        self.df_multilabel_labels = df_multilabel.iloc[:, 1:].to_numpy()
        self.df_multiclass_labels = df_multiclass.iloc[:, 1:].to_numpy()

        end = time.time()
        print("Preprocessing elapsed time: ", end - start, " seconds")
