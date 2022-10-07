from model import Model


model = Model(model_path='tf_models', ui_model=False)

model.train_models(epochs=500)
