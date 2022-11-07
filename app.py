from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from keras.models import load_model

model=load_model('model_flood.h5')
model.make_predict_function()

app = Flask(__name__)

def preprocess_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


def model_predict(img_path):
    preprocessed_image = preprocess_image(img_path)
    predictions = model.predict(preprocessed_image)
    result = np.argmax(predictions)
    labels = ['Flooding', 'No Flooding']
    return labels[result]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        result = model_predict(file_path)              
        return result
    return 'No'
if __name__ == '__main__':
    app.run(debug=True)
