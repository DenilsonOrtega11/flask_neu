from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64

import sys
print("Python version")
print(sys.version)


# Cargar el modelo
model = tf.keras.models.load_model('modelo.keras')

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files and 'camera' not in request.form:
        return redirect(request.url)

    file = request.files.get('file')
    img = None

    

    if file:
        img = Image.open(file.stream)
        img2 = img

    elif 'camera' in request.form:
        camera_image = request.form['camera']
        img = Image.open(io.BytesIO(base64.b64decode(camera_image.split(',')[1])))

    if img:
        
        img = img.resize((64, 64))  
        img2 = img
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        result = "El neumático está en buenas condiciones para ser usado." if predicted_class == 1 else "El neumático NO está en buenas condiciones."

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return render_template('index.html', prediction=result, image=img_base64)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
