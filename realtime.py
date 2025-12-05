import cv2
from ultralytics import YOLO
import time
import os

def run_realtime_tracking():
    # Ruta al modelo
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'overhead_people_tracking.pt')
    
    # Cargar el modelo
    try:
        print(f"Cargando modelo desde: {model_path}")
        model = YOLO(model_path)
        print("Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Captura de video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    window_name = "Real-Time People Tracking (YOLOv8)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    prev_frame_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("No se pudo leer el frame.")
            break

        results = model.track(frame, conf=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)
        annotated_frame = results[0].plot()

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time

        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detección finalizada.")

if __name__ == "__main__":
    run_realtime_tracking()
