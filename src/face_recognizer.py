from __future__ import annotations
import os
import pickle
from typing import List, Tuple, Optional
import numpy as np
from deepface import DeepFace
from deepface.commons import functions
from .utils import save_temp_frame

# Cosine distancia entre dos vectores de embedding. Devuelve un valor entre 0 y 1, donde 0 significa idéntico y 1 significa completamente diferente.
def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1.0
    return 1.0 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class FaceRecognizer:
    """
    Clase para manejar el reconocimiento facial utilizando DeepFace. Permite construir un índice de embeddings a partir de una base de datos de imágenes, y luego reconocer personas en frames capturados.
     - db_path: ruta a la carpeta raíz de la base de datos de imágenes (e.g. data/face_db/)
     - model_name: nombre del modelo de DeepFace a usar (e.g. "VGG-Face")
     - detector_backend: backend de detección de rostros (e.g. "mtcnn
     - build_index(persist_path): precomputar embeddings para todas las imágenes en db_path y guardarlos en memoria. Si persist_path es dado, también los guarda en un archivo pickle.
     - load_index(path): cargar un índice precomputado desde un archivo pickle.
     - recognize_frame(frame, threshold): reconocer una persona en un frame de OpenCV. 
      Devuelve (nombre o None, distancia). Una distancia más baja significa más similar (cosine -> 0 = idéntico). Si la distancia es mayor que el umbral, devuelve None.
    """



    def __init__(self, db_path: str = "data/face_db", model_name: str = "VGG-Face", detector_backend: str = "mtcnn"):
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.model = DeepFace.build_model(self.model_name)
        self.index: List[dict] = []

    def build_index(self, persist_path: str | None = None) -> None:
        """
        Precomputar embeddings para todas las imágenes en db_path y guardarlos en memoria. Si persist_path es dado, también los guarda en un archivo pickle.
         - persist_path: si se proporciona, ruta al archivo donde se guardará el índice precomputado (pickle). 
         Esto permite cargar el índice rápidamente en ejecuciones futuras sin tener que procesar todas las imágenes nuevamente.
         - El índice es una lista de diccionarios con keys: "name" (nombre de la persona), "path" (ruta a la imagen), "emb" (embedding vector).
         - El proceso ignora archivos que no son imágenes o que no pueden ser procesados por
         DeepFace, lo que permite tener una base de datos con archivos mixtos sin causar errores.
         - El índice se guarda en memoria en self.index para uso posterior en reconocimiento.
         - Ejemplo de uso:
              recognizer = FaceRecognizer(db_path="data/face_db")
              recognizer.build_index(persist_path="data/index.pkl")
          
        """
        entries = []
        for person in os.listdir(self.db_path):
            person_dir = os.path.join(self.db_path, person)
            if not os.path.isdir(person_dir):
                continue
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                try:
                    emb = DeepFace.represent(img_path=img_path, model=self.model, model_name=self.model_name, detector_backend=self.detector_backend, enforce_detection=False)
                    emb_arr = np.array(emb)[0] if isinstance(emb, list) and len(emb) > 0 else np.array(emb)
                    entries.append({"name": person, "path": img_path, "emb": emb_arr})
                except Exception:
                    continue
        self.index = entries
        if persist_path:
            with open(persist_path, "wb") as f:
                pickle.dump(self.index, f)

    def load_index(self, path: str) -> None:
        # Cargar un índice precomputado desde un archivo pickle. Esto permite cargar rápidamente un índice previamente construido sin tener que procesar todas las imágenes nuevamente.
        with open(path, "rb") as f:
            self.index = pickle.load(f)

    def recognize_frame(self, frame, threshold: float = 0.45) -> Tuple[Optional[str], float]:
        """
        Reconocer una persona en un frame de OpenCV. Devuelve (nombre o None, distancia). Una distancia más baja significa más similar (cosine -> 0 = idéntico). Si la distancia es mayor que el umbral, devuelve None.
         - frame: imagen capturada (array de OpenCV)
         - threshold: umbral de distancia para considerar un reconocimiento como válido (default: 0.45). Ajustar según el modelo y la calidad de las imágenes.
         - devuelve: (nombre o None, distancia)
        """
        tmp = save_temp_frame(frame)
        try:
            emb = DeepFace.represent(img_path=tmp, model=self.model, model_name=self.model_name, detector_backend=self.detector_backend, enforce_detection=False)
            emb_arr = np.array(emb)[0] if isinstance(emb, list) and len(emb) > 0 else np.array(emb)
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

        if not self.index:
            # Si el índice no está construido, construyámoslo ahora. Esto es útil para casos donde no se ha llamado explícitamente a build_index() antes de reconocer.
            self.build_index()

        best_name = None
        best_dist = float("inf")
        for e in self.index:
            d = _cosine_distance(emb_arr, np.array(e["emb"]))
            if d < best_dist:
                best_dist = d
                best_name = e["name"]

        if best_dist <= threshold:
            return best_name, float(best_dist)
        return None, float(best_dist)


__all__ = ["FaceRecognizer"]
