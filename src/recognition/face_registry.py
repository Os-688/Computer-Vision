from __future__ import annotations
import time
import os
from pathlib import Path
import cv2
from typing import List
from ..utils import safe_makedirs

# Recomendación: Si no consigues imagenes de calidad puedes tomar fotos desde tu teléfono móvil y transferirlas a la carpeta de la base de datos. 
# Asegúrate de que las imágenes sean claras, con buena iluminación y que el rostro esté centrado para obtener mejores resultados de reconocimiento.

#------------------------------------------------------------------
"""
Este modulo esta incorporado desde CLI register_face.py, que es una herramienta de línea de comandos
para registrar nuevas personas en la base de datos de rostros. Permite capturar imágenes desde la cámara,
guardarlas en la estructura de carpetas adecuada, y luego el sistema podrá reconocer esas personas durante la ejecución normal.


Comados de ejemplo para usar register_face.py:

python register_face.py --help # Muestra la ayuda con opciones disponibles
python register_face.py "Tu Nombre" 
python register_face.py "Tu Nombre" --images 8 --interval 0.7 # Captura 8 imágenes con un intervalo de 0.7 segundos entre cada una (ajusta según tu cámara y movimiento)
python register_face.py --list # Lista las personas actualmente registradas en la base de datos
python register_face.py --delete "Tu Nombre" # Elimina la carpeta de esa persona y sus imágenes (usa con precaución)

"""
# -----------------------------------------------------------------

# Función para capturar imágenes de una cámara y guardarlas en la base de datos de rostros. Se puede usar para agregar nuevas personas al sistema.

def capture_images_for_name(camera, name: str, n: int = 5, db_root: str | None = None, interval: float = 0.6) -> List[str]:
    """Captura n imagenes de la camara y las guarda en data/deepface/face_db/<name>/.

    Args:
        camera: Objeto de cámara con método get_frame() que devuelve (ok, frame).
        name: String con el nombre de la persona a capturar (usado para nombrar archivos y carpetas).
        n: Numero de imagenes a capturar (default: 5).
        db_root: ruta a la carpeta raíz de la base de datos (default: data/deepface/face_db). Si no se proporciona, se usará data/deepface/face_db.
        interval: segundos a esperar entre capturas (default: 0.6 segundos) para evitar capturar frames casi idénticos. Ajustar según la velocidad de movimiento y la tasa de frames de la cámara.

    Returns:
        Lista de rutas a las imágenes capturadas.
    """
    root = db_root or os.path.join("data", "deepface", "face_db")
    out_dir = Path(root) / name
    safe_makedirs(str(out_dir))
    saved: List[str] = []

    existing_indexes: List[int] = []
    for file_name in os.listdir(out_dir):
        stem = Path(file_name).stem
        if "_" in stem:
            suffix = stem.rsplit("_", 1)[-1]
            if suffix.isdigit():
                existing_indexes.append(int(suffix))

    start_index = max(existing_indexes) + 1 if existing_indexes else 0
    captured = 0

    while captured < n:
        ok, frame = camera.get_frame()
        if not ok or frame is None:
            print(f"  ⚠ Frame no disponible, esperando...")
            time.sleep(1.0)
            continue
        image_index = start_index + captured
        fname = out_dir / f"{name}_{image_index:03d}.jpg"
        # frame es esperado que sea un array de OpenCV, cv2.imwrite espera una ruta como string
        cv2.imwrite(str(fname), frame)
        saved.append(str(fname))
        print(f"  ✓ Imagen {captured + 1}/{n} capturada: {fname.name}")
        captured += 1
        time.sleep(interval)
    return saved


__all__ = ["capture_images_for_name"]
