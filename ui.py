"""
@author: enrico

"""

from flask import Flask, request, render_template
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
        if len(review.split()) == 0:
            return render_template('index.html', message='Please enter a review')
        prediction = model.predict_text(review)
        print(prediction)
        if prediction == {}:
            msg = 'No significant results'
        else:
            msg = 'The review tells about '
            cat = [c for c in prediction.keys()]
            sent = [s for s in prediction.values()]
            for i in range(len(prediction)):
                msg = msg + (cat[i] + ' in a ' + sent[i] + ' way, ')
            msg = msg[0:-2] + '.'
        return render_template('index.html', message=msg)


# Run Server
if __name__ == '__main__':
    app.run(debug=True)
