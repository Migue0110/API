
import os
import joblib
import gdown

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer


class Analizador:

    def __init__(self):

        # Directorio donde se encuentra este archivo
        self.base_path = Path(__file__).resolve().parent

        # Directorio de archivos entrenados
        self.training_path = self.base_path / "training"

        self.modelo_path = self.training_path / "modelo_random_forest.pkl"
        self.vocabulario_path = self.training_path / "vocabulario_countvectorizer.pkl"

        # Crear directorio si no existe
        self.training_path.mkdir(parents=True, exist_ok=True)

        # Descargar modelo si no existe
        if not self.modelo_path.exists():

            print("Descargando modelo...")

            gdown.download(
                id="1Vzhzqb-WqoR3l33Zvy5awwG3LAd-2WVT",
                output=str(self.modelo_path),
                quiet=False
            )

        # Verificar archivos necesarios
        if not self.modelo_path.exists():
            raise FileNotFoundError("No se encontró el modelo Random Forest.")

        if not self.vocabulario_path.exists():
            raise FileNotFoundError(
                "No se encontró vocabulario_countvectorizer.pkl. "
                "Debes incluirlo en el repositorio o descargarlo."
            )

        # Cargar archivos una sola vez
        print("Cargando modelo y vocabulario...")

        self.__model = joblib.load(self.modelo_path)
        self.__vocabulario = joblib.load(self.vocabulario_path)

        self.__vectorizer = CountVectorizer(
            vocabulary=self.__vocabulario
        )

        print("Analizador cargado correctamente.")

    def analizar(self, texto):

        if not texto or not texto.strip():
            raise ValueError("El texto no puede estar vacío.")

        mensaje = self.__vectorizer.transform([texto])

        prediccion = self.__model.predict(mensaje)

        if prediccion[0] == 1:
            return "Depresivo"

        elif prediccion[0] == 0:
            return "No depresivo"

        return "Resultado desconocido"
