from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    CAMERA_URL: str
    DB_PATH: str
    CSV_PATH: str
    MODEL_NAME: str
    DETECTOR_BACKEND: str
    THRESHOLD: float
    DEDUPE_SECONDS: int


def load_config() -> Config:
    """
    Carga la configuración desde variables de entorno o usa valores por defecto. Las variables de entorno permiten personalizar la configuración sin modificar el código, lo que es útil para despliegues o entornos de producción.
     - CAMERA_URL: URL de la cámara IP
     - DB_PATH: ruta a la base de datos de imágenes
     - CSV_PATH: ruta al archivo CSV de asistencia
     - MODEL_NAME: nombre del modelo de DeepFace a usar
     - DETECTOR_BACKEND: backend de detección de rostros a usar
     - THRESHOLD: umbral de distancia para reconocimiento facial (valores más bajos son más estrictos)
     - DEDUPE_SECONDS: número de segundos para considerar un registro de asistencia como duplicado
    """
    # Cargar variables de entorno con valores por defecto y convertir tipos según sea necesario
    # 
    # Usuario: para personalizar, puede crear un archivo .env con las variables deseadas o establecerlas en el entorno del sistema. Por ejemplo:
    #   CAMERA_URL=http://
    #   DB_PATH=data/deepface/face_db
    #   CSV_PATH=data/deepface/attendance/attendance.csv
    #   MODEL_NAME=Facenet512
    #   DETECTOR_BACKEND=retinaface
    #   THRESHOLD=0.4
    #   DEDUPE_SECONDS=300


    return Config(
        CAMERA_URL=os.getenv("CAMERA_URL", "http://192.168.1.100:8080/video"),
        DB_PATH=os.getenv("DB_PATH", os.path.join("data", "deepface", "face_db")),
        CSV_PATH=os.getenv("CSV_PATH", os.path.join("data", "deepface", "attendance", "attendance.csv")),
        MODEL_NAME=os.getenv("MODEL_NAME", "Facenet512"),
        DETECTOR_BACKEND=os.getenv("DETECTOR_BACKEND", "retinaface"),
        THRESHOLD=float(os.getenv("THRESHOLD", "0.4")),
        DEDUPE_SECONDS=int(os.getenv("DEDUPE_SECONDS", "300")),
    )


__all__ = ["Config", "load_config"]
