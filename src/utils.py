from __future__ import annotations
import os
import tempfile
from datetime import datetime
import typing as t
import cv2


"""
Utilidades generales para el proyecto de reconocimiento facial. Incluye funciones para manejar fechas, crear directorios de forma segura, y guardar frames temporales.
- now_date_str(): devuelve la fecha actual como string en formato YYYY-MM-DD.
- now_time_str(): devuelve la hora actual como string en formato HH:MM:SS.
- safe_makedirs(path): crea un directorio de forma segura, sin lanzar error si ya existe.
- save_temp_frame(frame): guarda un frame de OpenCV en un archivo temporal y devuelve la ruta al archivo. El archivo se crea con un sufijo .jpg y se borra automáticamente al cerrar el programa.
"""


def now_date_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def now_time_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_temp_frame(frame) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    # OpenCV expects str path
    cv2.imwrite(tmp_path, frame)
    return tmp_path


__all__ = ["now_date_str", "now_time_str", "safe_makedirs", "save_temp_frame"]
