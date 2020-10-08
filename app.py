from flask import Flask, render_template, redirect, request, url_for
import os
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2

model = vgg16.VGG16(include_top=True, weights="imagenet")

UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html", result="waiting for input")

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method=="POST":
        file = request.files["file"]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224,224))
        pred = model.predict([[img]])
        res = decode_predictions(pred)
        url = "5; url=https://www.google.com/search?q="+str(res[0][0][1])
        print(url)
        return render_template("index.html", result=str(res[0][0][1]), url=url)

if __name__ == "__main__":
    app.run()
