
from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


app = Flask(__name__)
model = VGG16()

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template("index.html")

@app.route('/', methods = ['POST'])
def predict():
    varImageFile = request.files['imageFile']
    imagePath = "./images/"+varImageFile.filename
    varImageFile.save(imagePath) 

    image = load_img(imagePath, target_size=(224,224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classifications = '%s (%.2f%%)' % (label[1], label[2]*100)

    return render_template("index.html", prediction = classifications)

if __name__ == '__main__' :
    app.run(port = 3000, debug = True)    