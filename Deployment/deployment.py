from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

# 모델 불러오기
model_path = '/Users/munpyeong-an/Downloads/model_17.ckpt'
model = load_model(model_path)

input_shape = model.layers[0].input_shape

app = FastAPI()

@app.get('/')
def root_route():
    return {"error": "use GET /prediction instead of root route"}

# 데이터 준비
@app.post('/prediction')
async def prediction_route(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = Image.open(BytesIO(contents))

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
    return {"result": int(prediction)}