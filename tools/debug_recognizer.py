"""Script de diagnóstico rápido para FaceRecognizer.

Uso:
  python tools/debug_recognizer.py [--sample PATH] [--backend opencv|mtcnn] [--model MODEL_NAME]

Si no se pasa --sample, el script tomará la primera imagen encontrada en la DB.

Imprime:
 - Número de embeddings construidos
 - Primeros matches (distancias) contra la imagen de prueba
 - Forma del embedding
"""

import argparse
import os
import sys

# Reducir ruido de logs de TensorFlow/oneDNN durante debugging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Asegurar que 'src' esté en path para imports locales
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import load_config
from face_recognizer import FaceRecognizer, _cosine_distance

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass


def find_first_image(db_path):
    for root, dirs, files in os.walk(db_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                return os.path.join(root, f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", help="Ruta a imagen de prueba (opcional)")
    parser.add_argument("--backend", default=None, help="Sobrescribir DETECTOR_BACKEND")
    parser.add_argument("--model", default=None, help="Sobrescribir MODEL_NAME")
    args = parser.parse_args()

    cfg = load_config()
    db_path = cfg.DB_PATH
    model_name = args.model or cfg.MODEL_NAME
    backend = args.backend or cfg.DETECTOR_BACKEND

    print(f"DB_PATH: {db_path}")
    print(f"Modelo: {model_name}")
    print(f"Detector backend: {backend}")

    recognizer = FaceRecognizer(db_path=db_path, model_name=model_name, detector_backend=backend)

    print("\nConstruyendo índice...")
    recognizer.build_index()
    print(f"Embeddings en índice: {len(recognizer.index)}")

    if len(recognizer.index) == 0:
        print("⚠ Índice vacío. Asegúrate de que hay imágenes en la ruta especificada.")
        return

    sample = args.sample
    if not sample:
        # Elegir primero una imagen que sí haya generado embedding en el índice
        sample = recognizer.index[0]["path"] if recognizer.index else find_first_image(db_path)
    if not sample:
        print("No se encontró imagen de prueba en la DB.")
        return

    print(f"\nUsando imagen de prueba: {sample}")

    # Probar con el recognizer directamente usando un frame
    try:
        import cv2
        import numpy as np
        
        # Cargar imagen como frame
        frame = cv2.imread(sample)
        if frame is None:
            print("⚠ No se pudo cargar la imagen")
            return
        
        # Usar el método del recognizer para obtener embedding
        emb_arr = recognizer._get_embedding(frame)
        method = "DeepFace"
        
        if emb_arr is not None and len(emb_arr) > 0:
            print(f"✓ Embedding extraído con {method}. Shape: {emb_arr.shape}")
        else:
            print(f"⚠ No se pudo extraer embedding con {method}")
            emb_arr = None
    except Exception as e:
        print(f"⚠ Error al extraer embedding de la imagen de prueba: {e}")
        import traceback
        traceback.print_exc()
        emb_arr = None

    if emb_arr is not None:
        # Calcular distancias con el índice y mostrar top-10
        dists = []
        for e in recognizer.index:
            d = _cosine_distance(emb_arr, e["emb"])
            dists.append((e["name"], e["path"], float(d)))
        dists = sorted(dists, key=lambda x: x[2])
        print("\nTop matches:")
        for name, path, d in dists[:10]:
            print(f"  {name} — {os.path.basename(path)} — distancia: {d:.4f}")

if __name__ == '__main__':
    main()
