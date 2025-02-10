import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
import gdown
import os

url = "https://drive.google.com/uc?id=1Vzhzqb-WqoR3l33Zvy5awwG3LAd-2WVT"
output_path = "modelo/training/modelo_random_forest.pkl"

if not os.path.exists(output_path):
    print("Descargando el modelo desde Google Drive...")
    gdown.download(url, output_path, quiet=False)



class Analizador():
    
    def analizar(self, texto):

        modelo_cargado = joblib.load(os.path.dirname(__file__) + '\\training\\modelo_random_forest.pkl')
        vocabulario_cargado = joblib.load(os.path.dirname(__file__) + '\\training\\vocabulario_countvectorizer.pkl')

        nuevo_vectorizer = CountVectorizer(vocabulary=vocabulario_cargado)
        mensaje = nuevo_vectorizer.transform([texto])
        prediccion = modelo_cargado.predict(mensaje)

        if mensaje == 1:
            return "Depresivo"
        elif mensaje == 0:
            return "No depresivo"