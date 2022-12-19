"""
@author: enrico

"""
from model import Model
import shutil

model = Model(model_path='tf_models', ui_model=False)

shutil.rmtree('tf_models', ignore_errors=True)

EPOCHS = 25

model.train_models(epochs=EPOCHS)
