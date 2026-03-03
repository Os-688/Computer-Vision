#!/usr/bin/env python3
"""
CLI para registrar nuevas personas en la base de datos de rostros.

Permite capturar imágenes desde la cámara IP, guardarlas y registrar personas
para que sean reconocidas por el sistema de asistencia.

Uso:
    python register.py "Tu Nombre"                           # Registrar persona
    python register.py "Tu Nombre" --images 8 --interval 0.7 # Con opciones
    python register.py --list                                # Listar registrados
    python register.py --delete "Tu Nombre"                  # Eliminar persona
    python register.py --help                                # Mostrar ayuda

Recomendación:
    Si no consigues imágenes de calidad, puedes tomar fotos desde tu teléfono
    móvil y transferirlas a la carpeta de la base de datos. Asegúrate de que
    las imágenes sean claras, con buena iluminación y el rostro centrado.
"""

import argparse
import os
import shutil
import time

from src.config import load_config
from src.camera import CameraIP
from src.recognition import capture_images_for_name


def count_images_in_dir(folder: str) -> int:
    """Cuenta imágenes en una carpeta."""
    if not os.path.isdir(folder):
        return 0
    return len(
        [
            name
            for name in os.listdir(folder)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
    )


def list_students(db_path: str) -> None:
    """Lista estudiantes registrados con cantidad de imágenes."""
    print("\n=== Estudiantes registrados ===")
    if not os.path.isdir(db_path):
        print("No existe la base de datos todavía.")
        return

    student_dirs = [
        name for name in sorted(os.listdir(db_path)) if os.path.isdir(os.path.join(db_path, name))
    ]

    if not student_dirs:
        print("No hay estudiantes registrados.")
        return

    for student in student_dirs:
        total = count_images_in_dir(os.path.join(db_path, student))
        print(f"- {student}: {total} imágenes")


def delete_student(db_path: str, name: str) -> None:
    """Elimina un estudiante y todas sus imágenes."""
    student_dir = os.path.join(db_path, name)
    if not os.path.isdir(student_dir):
        print(f"No existe el estudiante: {name}")
        return

    confirm = input(f"¿Eliminar '{name}' y sus imágenes? [s/N]: ").strip().lower()
    if confirm != "s":
        print("Operación cancelada.")
        return

    shutil.rmtree(student_dir)
    print(f"Estudiante eliminado: {name}")


def register_student(db_path: str, camera_url: str, name: str, images: int, interval: float) -> None:
    """Captura imágenes y registra un estudiante."""
    camera = CameraIP(camera_url=camera_url)
    try:
        camera.open()
        print(f"Conectado a cámara IP: {camera_url}")
        print(f"Capturando {images} imágenes para: {name}")
        
        # Warmup: esperar a que la cámara esté completamente lista
        print("Esperando estabilización de cámara...")
        warmup_attempts = 0
        max_warmup = 10
        while warmup_attempts < max_warmup:
            ok, frame = camera.get_frame()
            if ok and frame is not None:
                print(f"  ✓ Cámara estabilizada (frame {frame.shape})")
                break
            warmup_attempts += 1
            time.sleep(0.5)
        
        if warmup_attempts >= max_warmup:
            print("⚠ Advertencia: cámara tardó en estabilizar, continuando de todas formas...")
        
        print("¡Listo! Iniciando captura en 5 segundos...")
        time.sleep(5.0)
        
        saved = capture_images_for_name(
            camera=camera,
            name=name,
            n=images,
            db_root=db_path,
            interval=interval,
        )
        print(f"Captura finalizada. Imágenes guardadas: {len(saved)}")
        if saved:
            print(f"Carpeta: {os.path.dirname(saved[0])}")
    finally:
        camera.release()


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de CLI."""
    parser = argparse.ArgumentParser(
        description="CLI para registrar estudiantes en data/deepface/face_db usando cámara IP",
    )

    parser.add_argument(
        "name",
        nargs="?",
        help="Nombre del estudiante a registrar (ej. 'Juan Perez')",
    )
    parser.add_argument(
        "-i",
        "--images",
        type=int,
        default=5,
        help="Cantidad de imágenes a capturar (default: 5)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2,
        help="Segundos entre capturas (default: 1.5)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Listar estudiantes ya registrados",
    )
    parser.add_argument(
        "--delete",
        metavar="NOMBRE",
        help="Eliminar estudiante y sus imágenes",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Sobrescribir ruta de DB (por defecto usa DB_PATH de .env)",
    )
    parser.add_argument(
        "--camera-url",
        default=None,
        help="Sobrescribir URL de cámara (por defecto usa CAMERA_URL de .env)",
    )
    return parser


def main() -> None:
    """Función principal de la CLI de registro."""
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config()
    db_path = args.db_path or cfg.DB_PATH
    camera_url = args.camera_url or cfg.CAMERA_URL

    os.makedirs(db_path, exist_ok=True)

    if args.list:
        list_students(db_path)
        return

    if args.delete:
        delete_student(db_path, args.delete)
        return

    if not args.name:
        parser.error("Debes indicar un nombre o usar --list / --delete")

    if args.images < 1:
        parser.error("--images debe ser >= 1")

    register_student(
        db_path=db_path,
        camera_url=camera_url,
        name=args.name.strip(),
        images=args.images,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
