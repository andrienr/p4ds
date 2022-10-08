# Hotel reviews classification

## Description

The dataset, taken from here: http://sag.art.uniroma2.it/absita/data/, contains annotated hotel reviews scraped from the website booking.com
File dataviz.py contains some plots for exploring the dataset
The project performs two tasks:

- multilabel classification for detecting the review topic
- multiclass classification for detecting review polarity

Script used for the training process are preprocess.py and model.py
Script used for the user interface is ui.py
Directory `tf_models contains` contains tensorflow models for both multilabel and multiclass models, trained for 500 epochs
Directory `confusion_matrices` contains confusion matrix plots for all the 8 categories, generated during the evaluation phase
A small web app in Flask is user for trying out the model predictions

## Usage

```
pip install -r requirements.txt
python3 ui.py
```

`preprocess.py` creates 2 csv files and saves them to `data` directory for using in the data visualization jupyter notebook

`train.py` will train the multilabel model and the multiclass models for each of the 8 categories, and save them to `tf_models` directory as tensorflow models for serving

`evaluate.py` will evaluate through cross validation procedure the models accuracy and plot confusion matrices graphs

If the folder `tf_models` does not exist, ui.py will train first the model for few epochs as well
