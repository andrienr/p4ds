from model import Model
import numpy as np
from sklearn.preprocessing import OneHotEncoder


model = Model(model_path='tf_models', ui_model=False)

# evaluate multilabel model
multilabel_results = model.evaluate_model(
    model.features, model.df_multilabel_labels, 'multilabel', epochs=100)

# evaluate multiclass model
for i in range(len(model.categories)):
    multiclass_results = model.evaluate_model(
        features=model.features, labels=model.df_multiclass_labels[:, i],
        task='multiclass', epochs=100, cat=i)

# summarize performances
print('Multilabel model accuracy: %.3f (%.3f)' %
      (np.mean(multilabel_results), np.std(multilabel_results)))
print('Multiclass model accuracy: %.3f (%.3f)' %
      (np.mean(multiclass_results), np.std(multiclass_results)))
