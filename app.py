from flask import Flask, render_template, request, url_for
from keras.models import load_model
import cv2
import numpy as np


app = Flask(__name__)

model = load_model('covid-19_prediction.h5')
print("Model is loaded")


@app.route('/')
def index():
    return render_template('index.html', data="HELLO")


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    img = request.files['img']
    img.save('static/img.jpg')

    img = cv2.imread('static/img.jpg')
    
    tmp = img
    tmp = cv2.resize(tmp, (400, 400))
    cv2.imwrite('static/img.jpg', tmp)


    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_NEAREST)
    img = img / 255.0

    class_names = ['COVID NEGATIVE', 'COVID POSITIVE']
    prediction = model.predict(np.array([img]))
    predicted_class = class_names[np.argmax(prediction)]

    return render_template("prediction.html", data=predicted_class)


if __name__ == '__main__':
    app.run(debug=True)
