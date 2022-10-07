"""
@author: enrico

"""

from flask import Flask,  request, render_template
import os
from model import Model


# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

model = Model(model_path='tf_models')


@app.route('/')
def index():
    return render_template('index.html')


@app.route(
    '/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        review = request.form['review']
        prediction = model.predict_text(review)
        # prediction = review
        return render_template('index.html', message=prediction)


# Run Server
if __name__ == '__main__':
    app.run(debug=True)
