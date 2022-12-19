"""
@author: enrico

"""
from model import Model
import numpy as np
import shutil

model = Model(model_path='tf_models', ui_model=False)

shutil.rmtree('confusion_matrices', ignore_errors=True)

EPOCHS = 15
N_SPLITS = 5

# evaluate multilabel model
multilabel_results = model.evaluate_model(
    model.features, model.df_multilabel_labels, 'multilabel', epochs=EPOCHS, n_splits=N_SPLITS)
# summarize performance
print('Multilabel model accuracy: %.3f (%.3f)' %
      (np.mean(multilabel_results), np.std(multilabel_results)))

# evaluate multiclass model
for i in range(len(model.categories)):
    multiclass_results = model.evaluate_model(
        features=model.features, labels=model.df_multiclass_labels[:, i],
        task='multiclass', epochs=EPOCHS, n_splits=N_SPLITS, cat=i)
    # summarize performance
    print('Multiclass model accuracy - ' + str(model.categories[i]) + ': %.3f (%.3f)' %
          (np.mean(multiclass_results), np.std(multiclass_results)))
