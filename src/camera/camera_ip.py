from __future__ import annotations
import time
import typing as t
import cv2

from .camera_manager import CamaraManager


class CameraIP:
    """
    Envoltura para manejar unicamente cámaras IP (No locales) con reintentos automáticos en caso de fallos de conexión.
     - camera_url: URL de la cámara IP (e.g. rtsp://user:pass@ip:port/stream)
     - reopen_backoff: segundos a esperar antes de intentar reconectar tras un fallo
     - get_frame() intentará obtener un frame, y si falla, hará un intento de reconexión antes de devolver False.
     - release() cerrará la conexión a la cámara.

    """
    def __init__(self, camera_url: str, reopen_backoff: float = 1.0):
        clean_url = (camera_url or "").strip().strip('"').strip("'")
        if not clean_url:
            raise ValueError("camera_url está vacío. Revisa CAMERA_URL en el archivo .env")
        self.camera_url = clean_url
        self._mgr = CamaraManager(ip_url=clean_url)
        self.reopen_backoff = reopen_backoff

    def open(self) -> None:
        self._mgr.open()

    def get_frame(self) -> t.Tuple[bool, t.Optional["cv2.Mat"]]:
        try:
            ok, frame = self._mgr.get_frame()
            if not ok:
                # attempt quick reconnect
                try:
                    self._mgr.release()
                except Exception:
                    pass
                time.sleep(self.reopen_backoff)
                self._mgr.open()
                return self._mgr.get_frame()
            return ok, frame
        except Exception:
            # backoff and try to re-open
            try:
                self._mgr.release()
            except Exception:
                pass
            time.sleep(self.reopen_backoff)
            try:
                self._mgr.open()
                return self._mgr.get_frame()
            except Exception:
                return False, None

    def release(self) -> None:
        self._mgr.release()


__all__ = ["CameraIP"]
