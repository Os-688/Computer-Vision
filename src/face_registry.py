from __future__ import annotations
import time
import os
from pathlib import Path
import cv2
from typing import List
from .utils import safe_makedirs


def capture_images_for_name(camera, name: str, n: int = 5, db_root: str | None = None, interval: float = 0.6) -> List[str]:
    """Capture n images from the camera and save them in data/face_db/<name>/.

    Args:
        camera: Objeto de cámara con método get_frame() que devuelve (ok, frame).
        name: String con el nombre de la persona a capturar (usado para nombrar archivos y carpetas).
        n: Numero de imagenes a capturar (default: 5).
        db_root: ruta a la carpeta raíz de la base de datos (default: data/face_db). Si no se proporciona, se usará data/face_db.
        interval: segundos a esperar entre capturas (default: 0.6 segundos) para evitar capturar frames casi idénticos. Ajustar según la velocidad de movimiento y la tasa de frames de la cámara.

    Returns:
        Lista de rutas a las imágenes capturadas.
    """
    root = db_root or os.path.join("data", "face_db")
    out_dir = Path(root) / name
    safe_makedirs(str(out_dir))
    saved: List[str] = []
    i = 0
    while i < n:
        ok, frame = camera.get_frame()
        if not ok or frame is None:
            time.sleep(0.5)
            continue
        fname = out_dir / f"{name}_{i:03d}.jpg"
        # frame es esperado que sea un array de OpenCV, cv2.imwrite espera una ruta como string
        cv2.imwrite(str(fname), frame)
        saved.append(str(fname))
        i += 1
        time.sleep(interval)
    return saved


__all__ = ["capture_images_for_name"]
