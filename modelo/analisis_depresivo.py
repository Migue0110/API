import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
import gdown

class Analizador():
        
    def __init__(self):
        self.__model = None
        self.modelo_path = "/opt/render/project/src/modelo/training/modelo_random_forest.pkl"

        if not os.path.exists(self.modelo_path):
            print("Descargando modelo...")
            gdown.download("https://drive.google.com/uc?id=1Vzhzqb-WqoR3l33Zvy5awwG3LAd-2WVT", self.modelo_path, quiet=False)
            print("Modelo descargado.")
                
    def analizar(self, texto):
        modelo_path = os.path.join(os.path.dirname(__file__), 'training', 'modelo_random_forest.pkl')
        vocabulario_path = os.path.join(os.path.dirname(__file__), 'training', 'vocabulario_countvectorizer.pkl')

        modelo_cargado = joblib.load(modelo_path)
        vocabulario_cargado = joblib.load(vocabulario_path)

        if vocabulario_cargado is None:
            raise ValueError("El vocabulario no se cargó correctamente.")

        nuevo_vectorizer = CountVectorizer(vocabulary=vocabulario_cargado)
        mensaje = nuevo_vectorizer.transform([texto])
        prediccion = modelo_cargado.predict(mensaje)

        # 🛠 CORRECCIÓN DEL ERROR 🛠
        if prediccion[0] == 1:
            return "Depresivo"
        elif prediccion[0] == 0:
            return "No depresivo"
