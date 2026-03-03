#!/usr/bin/env python3
"""
Sistema de Asistencia con Reconocimiento Facial - Punto de entrada principal.

Detecta rostros desde una cámara IP y registra la asistencia en tiempo real.

Uso:
    python main.py

Configuración:
    - Edita .env con tu URL de cámara IP y parámetros del modelo
    - Primera ejecución descargará modelos de DeepFace (~200MB)

Controles:
    - Presiona 'q' o ESC para salir
"""

import os
import sys
import time
import logging
from datetime import datetime

# Suprimir logs de TensorFlow/DeepFace para consola más limpia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('deepface').setLevel(logging.ERROR)

import cv2

from src.config import load_config
from src.camera import CameraIP
from src.recognition import FaceRecognizer
from src.attendance import AttendanceService

def main():
    # cargar configuración
    cfg = load_config()
    cfg.CAMERA_URL = (cfg.CAMERA_URL or "").strip().strip('"').strip("'")

    print("=" * 60)
    print("Sistema de Asistencia con Reconocimiento Facial")
    print("=" * 60)
    print(f"Cámara IP: {cfg.CAMERA_URL}")
    print(f"Base de datos: {cfg.DB_PATH}")
    print(f"CSV asistencia: {cfg.CSV_PATH}")
    print("=" * 60)

    # inicializar componentes
    cam = CameraIP(camera_url=cfg.CAMERA_URL)
    # inicializar reconocedor con opciones de modelo y detector basados en configuración, con valores por defecto
    recognizer = FaceRecognizer(
        db_path=cfg.DB_PATH,
        model_name=cfg.MODEL_NAME,
        detector_backend=cfg.DETECTOR_BACKEND,
    )
    # attendance service con dedupe opcional basado en configuración
    attendance = AttendanceService(cfg.CSV_PATH, dedupe_seconds=cfg.DEDUPE_SECONDS)

    # intentar construir índice de embeddings (si el recognizer lo soporta)
    print("\nConstruyendo índice de rostros conocidos...")
    print("(Primera ejecución puede tardar mientras se descargan modelos...)")
    try:
        recognizer.build_index()
        if not recognizer.index:
            print("⚠ ADVERTENCIA: El índice está vacío. Verifica que existan imágenes en data/deepface/face_db/")
        else:
            print(f"✓ Índice construido: {len(recognizer.index)} personas en la base de datos")
    except Exception as e:
        print(f"⚠ Error construyendo índice: {e}")
        print("Continuando sin índice preconstruido")
    
    # abrir cámara con manejo de excepciones
    print(f"\nConectando a cámara: {cfg.CAMERA_URL}")
    try:
        cam.open()
        print("✓ Cámara conectada\n")
    except Exception as e:
        print(f"\n❌ ERROR: No se pudo abrir la cámara")
        print(f"Detalles: {e}")
        print("\nVerifica:")
        print("1. Que el celular esté en la misma red WiFi")
        print("2. Que la app de cámara IP esté ejecutándose")
        print("3. Que la URL sea correcta en el archivo .env")
        print("4. Prueba abrir la URL en tu navegador primero")
        return
    
    window_name = "Asistencia - Presiona 'q' para salir"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Sistema iniciado. Presiona 'q' para salir.\n")

    # loop principal con manejo de errores para reconexión y reconocimiento
    reconnect_attempts = 0
    max_reconnect = 5
    
    try:
        while True:
            ok, frame = cam.get_frame()
            if not ok or frame is None:
                reconnect_attempts += 1
                if reconnect_attempts > max_reconnect:
                    print(f"\n❌ Conexión perdida después de {max_reconnect} intentos")
                    break
                    
                print(f"⚠ Frame no disponible, reintentando ({reconnect_attempts}/{max_reconnect})...")
                
                # liberar y reintentar
                try:
                    cam.release()
                except Exception:
                    pass
                    
                time.sleep(2.0)
                
                try:
                    cam.open()
                    print("✓ Reconectado")
                    reconnect_attempts = 0  # resetear contador si reconecta
                except Exception as e:
                    print(f"Error al reconectar: {e}")
                    time.sleep(1.0)
                    
                continue
            
            # resetear contador si frame exitoso
            reconnect_attempts = 0

            # reconocimiento (manejar excepciones internas)
            try:
                name, score = recognizer.recognize_frame(frame, threshold=cfg.THRESHOLD)
            except Exception as e:
                print(f"⚠ Error en reconocimiento: {e}")
                name, score = None, None

            # marcar asistencia si hay match
            if name:
                try:
                    marked = attendance.mark_attendance(name)
                    if marked:
                        print(f"✓ Asistencia registrada: {name} ({datetime.now().strftime('%H:%M:%S')})")
                except Exception as e:
                    print(f"Error al marcar asistencia: {e}")

            # dibujar info en frame
            label = f"{name}" if name else "Desconocido"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Fondo para mejor legibilidad del nombre
            cv2.rectangle(frame, (5, 5), (min(400, frame.shape[1]-5), 50), (0, 0, 0), -1)
            color = (0, 255, 0) if name else (0, 0, 255)
            cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            
            # Timestamp abajo
            cv2.rectangle(frame, (5, frame.shape[0] - 35), (min(350, frame.shape[1]-5), frame.shape[0] - 5), (0, 0, 0), -1)
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\nCerrando sistema...")
                break

    except KeyboardInterrupt:
        print("\n\nInterrumpido por usuario")
    finally:
        try:
            cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Sistema cerrado")

if __name__ == "__main__":
    main()
