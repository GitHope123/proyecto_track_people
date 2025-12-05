import cv2
from ultralytics import YOLO
import time

def run_realtime_tracking():
    # Rutas
    model_path = '/home/hugo/proyecto_track_people/models/overhead_people_tracking.pt'
    
    # Cargar el modelo entrenado
    try:
        print(f"Cargando modelo desde: {model_path}")
        model = YOLO(model_path)
        print("Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Inicializar la captura de video (0 para la webcam por defecto)
    # Si tienes un archivo de video, cambia el 0 por la ruta del archivo, ej: 'video.mp4'
    video_source = 0 
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("No se pudo abrir la cámara o el archivo de video.")
        return

    print("Iniciando detección en tiempo real... Presiona 'q' para salir.")
    
    # Configuración de ventana
    window_name = "Real-Time People Tracking (YOLOv8)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Variables para FPS
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("No se pudo leer el frame (¿Fin del video?).")
            break

        # Realizar inferencia y rastreo
        # conf=0.5: umbral de confianza
        # persist=True: necesario para que el tracking funcione entre frames
        results = model.track(frame, conf=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)

        # Dibujar los resultados en el frame
        annotated_frame = results[0].plot()

        # Cálculo de FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        
        # Mostrar FPS en pantalla
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar el frame
        cv2.imshow(window_name, annotated_frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Detección finalizada.")

if __name__ == "__main__":
    run_realtime_tracking()
