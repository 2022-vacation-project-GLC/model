from flask import Flask, render_template,request
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np


model_path = '/Users/bandaejun/Documents/GitHub/model/Deployment/model_80.ckpt'
model = load_model(model_path)

input_shape = model.layers[0].input_shape

app = Flask(__name__)
image_path = '/Users/bandaejun/Documents/GitHub/model/Deployment/images'
app.config["UPLOAD_FOLDER"] = image_path


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file1" not in request.files:
            return "there is no file1 in form!"
        file1 = request.files["file1"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
        file1.save(path)
        pil_image = Image.open(path)
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
        prediction = int(np.around(predictions[0]))
        # return str(prediction)
        if prediction ==1:
            return"""
        <h1>Result of the Input Image</h1>
            <p>There is no Gravitational Lensing Effect</p>
        """
        else:
            return"""
        <h1>Result of the Input Image</h1>
            <p>There is no Gravitational Lensing Effect</p>
        """  
        # else:
        #     return 
        
    return """
    <h1>Binary Classification about Gravitational Lensing Effect</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit" value='Predict Image'>
    </form>
    """
    

if __name__ =='__main__':
    app.run(port=3000,debug=True)