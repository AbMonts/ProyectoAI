import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .ml_model import classify_image  # Importa la función de clasificación desde tu módulo de modelo de IA
import os

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('../covid-19/modelo_covid19.h5')

# Función para preprocesar la imagen antes de hacer la predicción
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)  # Convertir la imagen a array
    img_array = np.expand_dims(img_array, axis=0)  # Expande las dimensiones para hacerla compatible con el modelo
    img_array /= 255.0  # Normalizar los valores de píxeles
    return img_array

def predict_image(request):
    if request.method == "POST":
        image = request.FILES['image']
        
        # Usa FileSystemStorage para manejar el almacenamiento de archivos
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'subidos'))
        filename = fs.save(image.name, image)
        img_path = fs.path(filename)

        # Realiza el preprocesamiento y clasificación de la imagen
        img_array = preprocess_image(img_path)  # Función para preparar la imagen para el modelo
        result = classify_image(img_array)  # Clasifica la imagen

        # Genera la URL de la imagen para mostrarla en la plantilla
        img_url = os.path.join(settings.MEDIA_URL, 'subidos', image.name)

        # Renderiza la plantilla con el resultado y la URL de la imagen
        return render(request, 'resultado.html', {'result': result, 'img_url': img_url})

    return render(request, 'clasificacion/cargar_imagen.html') ##puede ser mejor classify
