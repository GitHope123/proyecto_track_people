"""
🎥 Detección de Personas en Tiempo Real - Vista Cenital
Modelo: YOLOv8n entrenado con 5,107 imágenes overhead
"""

import cv2
from ultralytics import YOLO
import time
from pathlib import Path

# ==============================
# CONFIGURACIÓN
# ==============================

MODEL_PATH = 'overhead.pt'
CONFIDENCE = 0.25      # Umbral de confianza (ajustar según necesidad)
CAMERA_ID = 0          # 0 = cámara predeterminada, 1 = segunda cámara, etc.
WINDOW_NAME = 'Detección de Personas - Vista Cenital'

# Configuración de visualización
SHOW_FPS = True
SHOW_CONFIDENCE = True
BOX_COLOR = (0, 255, 0)      # Verde (BGR)
TEXT_COLOR = (255, 255, 255)  # Blanco
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ==============================
# VERIFICAR MODELO
# ==============================

if not Path(MODEL_PATH).exists():
    print(f"❌ Error: No se encontró el modelo en '{MODEL_PATH}'")
    print("💡 Asegúrate de que 'best.pt' esté en la misma carpeta que main.py")
    exit(1)

print("✅ Modelo encontrado")
print(f"📁 Ruta: {Path(MODEL_PATH).absolute()}")

# ==============================
# CARGAR MODELO
# ==============================

print("\n🤖 Cargando modelo YOLOv8n...")
model = YOLO(MODEL_PATH)
print("✅ Modelo cargado exitosamente")

# ==============================
# INICIALIZAR CÁMARA
# ==============================

print(f"\n🎥 Iniciando cámara {CAMERA_ID}...")
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print(f"❌ Error: No se pudo abrir la cámara {CAMERA_ID}")
    print("💡 Verifica que:")
    print("   • La cámara esté conectada")
    print("   • No esté siendo usada por otra aplicación")
    print("   • Tienes permisos de acceso a la cámara")
    exit(1)

# Configurar resolución (opcional, ajustar según tu cámara)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_camera = int(cap.get(cv2.CAP_PROP_FPS))

print(f"✅ Cámara iniciada:")
print(f"   • Resolución: {width}x{height}")
print(f"   • FPS cámara: {fps_camera}")
print(f"\n🚀 Presiona 'q' o 'ESC' para salir")
print("=" * 60)

# ==============================
# LOOP PRINCIPAL
# ==============================

fps_time = time.time()
fps_counter = 0
fps_display = 0

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️ Error al leer frame de la cámara")
            break
        
        # Realizar detección
        results = model(frame, conf=CONFIDENCE, verbose=False)
        
        # Procesar resultados
        detections = results[0].boxes
        num_persons = len(detections)
        
        # Dibujar detecciones
        for box in detections:
            # Coordenadas del bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Dibujar rectángulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            
            # Dibujar confianza si está habilitado
            if SHOW_CONFIDENCE:
                label = f'Persona {conf:.2f}'
                label_size, _ = cv2.getTextSize(label, FONT, 0.5, 1)
                
                # Fondo para el texto
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    BOX_COLOR,
                    -1
                )
                
                # Texto
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    FONT,
                    0.5,
                    TEXT_COLOR,
                    1
                )
        
        # Calcular FPS real
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        # Información en pantalla
        info_y = 30
        cv2.putText(
            frame,
            f'Personas detectadas: {num_persons}',
            (10, info_y),
            FONT,
            0.7,
            (0, 255, 255),
            2
        )
        
        if SHOW_FPS:
            cv2.putText(
                frame,
                f'FPS: {fps_display}',
                (10, info_y + 35),
                FONT,
                0.7,
                (0, 255, 255),
                2
            )
        
        # Mostrar frame
        cv2.imshow(WINDOW_NAME, frame)
        
        # Salir con 'q' o 'ESC'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n👋 Cerrando aplicación...")
            break

except KeyboardInterrupt:
    print("\n⚠️ Interrupción del usuario")

except Exception as e:
    print(f"\n❌ Error inesperado: {e}")

finally:
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Recursos liberados correctamente")
    print("=" * 60)