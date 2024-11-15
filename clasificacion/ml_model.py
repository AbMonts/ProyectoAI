from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model = load_model('../covid-19/modelo_covid19.h5')
def classify_image(img_file):
    img = image.load_img(img_file, target_size=(224, 224))  # Ajusta el tamaño según tu modelo
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    return "Con COVID-19" if result[0][0] > 0.5 else "Sin COVID-19"
