# Hotel reviews classification

## Description

The project performs two tasks:

- multilabel classification for detecting the review topic
- multiclass classification for detecting review polarity

Both models are composed of 2 dense neural network layers

They are trained on a dataset, taken from http://sag.art.uniroma2.it/absita/data/, that contains annotated hotel reviews scraped from the website booking.com

File `dataviz.py` contains some plots for exploring the dataset

Script used for the training process are `preprocess.py` and `model.py`

Script used for the user interface is `ui.py`

Folder `tf_models` contains tensorflow models for both multilabel and multiclass models, trained for 500 epochs

Folder `confusion_matrices` contains confusion matrix plots for all the 8 categories, generated during the evaluation phase

A small web app in Flask can be used for trying out the model predictions

## Usage

```
pip install -r requirements.txt
python3 ui.py
```

`preprocess.py` creates 2 csv files and saves them to `data` folder for using in the data visualization jupyter notebook

`train.py` will train the multilabel model and the multiclass models for each of the 8 categories, and save them to `tf_models` folder as tensorflow models for serving

`evaluate.py` will evaluate through cross validation procedure the models accuracy and plot confusion matrices graphs

If the folder `tf_models` does not exist, `ui.py` will train first the model for few epochs as well
