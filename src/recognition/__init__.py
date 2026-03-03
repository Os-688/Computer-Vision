"""
Módulo de reconocimiento facial.
"""

from .face_recognizer import FaceRecognizer
from .face_registry import capture_images_for_name

__all__ = ["FaceRecognizer", "capture_images_for_name"]
