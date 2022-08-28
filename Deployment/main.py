from flask import Flask, render_template,request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


model_path = '/Users/bandaejun/Documents/GitHub/model/Deployment/model_80.ckpt'
model = load_model(model_path)

input_shape = model.layers[0].input_shape

app = Flask(__name__)

@app.route('/',methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path="./uploads/"+imagefile.filename
    imagefile.save(image_path)
    
    pil_image = Image.open(image_path)
    pil_image = pil_image.resize((101,101))
    pil_image = pil_image.convert('L')

    numpy_array = np.array(pil_image)
    numpy_array = np.expand_dims(numpy_array, axis=-1)
    numpy_array = numpy_array / 255.0
    if numpy_array.sum() > 200:
        numpy_array = 1 - numpy_array
    prediction_array = np.array([numpy_array])

	# 예측 및 반환
    predictions = model.predict(prediction_array)
    prediction = np.around(predictions[0])
    return int(prediction)
    

if __name__ =='__main__':
    app.run(port=3000,debug=True)