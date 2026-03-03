# Proyecto: Sistema de asistencia con reconocimiento facial

Prototipo local en Python con DeepFace + OpenCV para registrar asistencia desde una cámara IP (celular en la misma red WiFi).

**Requisitos previos:**
- Python 3.8+
- Conexión WiFi estable
- Celular con app de cámara IP (ej: IP Webcam, DroidCam, iVCam)

## 1) Preparación del entorno

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Configuración

**⚠️ Nota importante - Permisos de cámara:**
Asegúrate de que la app de cámara IP del celular tenga permisos habilitados:
- Android: Permite acceso a cámara y micrófono
- iOS: Verifica Configuración > Privacidad > Cámara

Edita el archivo `.env` con tu URL de cámara IP:

```env
CAMERA_URL=http://TU_IP:8080/video
DB_PATH=data/deepface/face_db
CSV_PATH=data/deepface/attendance/attendance.csv
MODEL_NAME=VGG-Face
DETECTOR_BACKEND=opencv
THRESHOLD=0.4
DEDUPE_SECONDS=300
```

## 3) Registrar estudiantes (CLI)

Registrar una persona:

```powershell
python register.py "Tu Nombre"
```

Opciones útiles:

```powershell
python register.py --list
python register.py --delete "Tu Nombre"
python register.py "Tu Nombre" --images 8 --interval 0.7
```

Las imágenes se guardan en:

`data/deepface/face_db/<Nombre>/`

**Opción Manual - Colocar imágenes directamente:**

Crea una carpeta con el nombre de la persona en:
```
data/deepface/face_db/<Nombre>/
```
**Requisitos de las imágenes:**
- **Formato**: JPG o PNG
- **Cantidad**: Mínimo 10, óptimo 15-20 imágenes
- **Resolución**: Mínimo 100x100 píxeles (recomendado 300x300+)
- **Ángulos**: Frontal, 45°, perfil, arriba, abajo (variable)
- **Iluminación**: Clara y uniforme, evitar sombras en rostro
- **Fondo**: Neutro preferiblemente

**Estructura:**
```
data/deepface/face_db/
├── Juan Pérez/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── María López/
    ├── img_001.jpg
    └── ...
```

**Validación**: La próxima vez que ejecutes `python main.py`, las imágenes se cargarán automáticamente en el índice.

## 4) Ejecutar asistencia en tiempo real

```powershell
python main.py
```

El sistema reconoce rostros desde la base y registra asistencia en `data/deepface/attendance/attendance.csv`.

**Formato del archivo CSV:**
```
date,time,name,status
2026-03-02,14:30:45,Juan Pérez,present
2026-03-02,14:31:12,María López,present
2026-03-02,14:32:00,Juan Pérez,present
```

## 5) Solución de Problemas

**❌ Error: "No se pudo abrir la cámara"**
- Verifica que el celular esté en la misma red WiFi que la computadora
- Asegúrate de que la URL en `.env` es correcta (prueba abrirla en el navegador primero)
- Reinicia la app de cámara IP del celular
- Comprueba que no hay firewall bloqueando la conexión

**❌ Error: "El índice está vacío"**
- Verifica que hay imágenes en `data/deepface/face_db/<Nombre>/`
- Asegúrate de que las imágenes cumplen los requisitos de calidad
- Intenta registrar manualmente con `python register.py "Tu Nombre"`

**❌ Reconocimiento muy lento**
- Primera ejecución descarga modelos (~200MB) - esperado
- Reduce la resolución de la cámara si es posible
- Intenta con un modelo más ligero: `MODEL_NAME=MobileNet` en `.env`

**❌ Falsos positivos (reconoce personas equivocadas)**
- Aumenta el umbral en `.env`: `THRESHOLD=0.5` (más estricto)
- Agrega más imágenes de entrenamiento por persona (15-20)
- Usa mejor iluminación al capturar imágenes