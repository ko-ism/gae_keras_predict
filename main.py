import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from wtforms import Form, validators, ValidationError, SelectField
from google.cloud import storage
import numpy as np
from keras.models import load_model
from PIL import Image


PROJECT_ID = '<change to your project id>'
CLOUD_STORAGE_BUCKET = '<change to your bucket>'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])
DOWNLOAD_FOLDER = '/tmp/'

model_name = 'nakamoto_jiro_cnn.h5'
classes = ["nakamoto","jiro"]
num_classes = len(classes)
image_size = 50

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/post_image", methods=['POST'])
def post_image():
    try:
        predict_kekka = ""
        file = request.files['file_predict']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
        file.save(os.path.join(DOWNLOAD_FOLDER , filename))
        filepath = os.path.join(DOWNLOAD_FOLDER , filename)
        modelpath = os.path.join(DOWNLOAD_FOLDER , model_name)

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
        blob = bucket.get_blob(model_name)
        blob.download_to_filename(modelpath)
        model = load_model(modelpath)

        image = Image.open(filepath)
        image = image.convert('RGB')
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X = []
        X.append(data)
        X = np.array(X)

        result = model.predict([X])[0]
        predicted = result.argmax()
        percentage = int(result[predicted] * 100)

        predict_kekka = "ラベル： " + classes[predicted] + ", 確率："+ str(percentage) + " %"
        
        message = "success!"
        
        return render_template('predict_kekka.html', predict_kekka=predict_kekka, message=message)
    except Exception as e:
        message = "false"+str(e)
        return render_template('predict_kekka.html', message=message)

    return 'OK'

