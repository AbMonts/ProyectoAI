import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('../covid-19/modelo_covid19.h5')

# Función para preprocesar la imagen antes de hacer la predicción
def preprocess_image(img_path):
    img = image.image_utils.load_img(img_path, target_size=(150, 150))
    img_array = image.image_utils.img_to_array(img)  # Convertir la imagen a array
    img_array = np.expand_dims(img_array, axis=0)  # Expande las dimensiones para hacerla compatible con el modelo
    img_array /= 255.0  # Normalizar los valores de píxeles
    return img_array

# Función para clasificar la imagen
def classify_image(img_array):
    # Realiza la predicción
    predictions = model.predict(img_array)
    # Interpreta el resultado de la predicción
    if predictions[0] < 0.5:
        return "Sin COVID-19"
    else:
        return "Con COVID-19"

# Vista para manejar la carga y clasificación de la imagen
def predict_image(request):
    if request.method == "POST":
        image_file = request.FILES['image']
        
        # Usa FileSystemStorage para manejar el almacenamiento de archivos
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'subidos'))
        filename = fs.save(image_file.name, image_file)
        img_path = fs.path(filename)

        # Realiza el preprocesamiento y clasificación de la imagen
        img_array = preprocess_image(img_path)
        result = classify_image(img_array)  # Clasifica la imagen

        # Genera la URL de la imagen para mostrarla en la plantilla
        img_url = os.path.join(settings.MEDIA_URL, 'subidos', image_file.name)

        # Renderiza la plantilla con el resultado y la URL de la imagen
        return render(request, 'clasificacion/resultado.html', {'result': result, 'img_url': img_url})

    return render(request, 'clasificacion/cargar_imagen.html')
