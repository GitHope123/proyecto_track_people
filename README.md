# 🚶‍♂️ Overhead People Tracking - YOLOv8

Este proyecto implementa un sistema de **detección y seguimiento de personas** en tiempo real utilizando **YOLOv8**, optimizado para imágenes tomadas desde una perspectiva cenital (overhead).

El modelo ha sido entrenado durante 50 épocas, logrando un rendimiento excepcional con un **mAP@50 del 98.3%**.

## 📂 Estructura del Proyecto

El repositorio contiene únicamente los archivos esenciales para el funcionamiento y re-entrenamiento del modelo:

```bash
proyecto_track_people/
├── models/
│   └── overhead_people_tracking.pt   # 🧠 Modelo entrenado (Mejores pesos)
├── notebooks/
│   ├── entrenamiento_yolo.ipynb      # 📓 Notebook para entrenar el modelo desde cero
│   └── clean_images.py               # 🧹 Script para limpieza de datos (eliminar duplicados/corruptos)
├── realtime.py                       # 📹 Script para detección y tracking en tiempo real
├── requirements.txt                  # 📦 Dependencias del proyecto
└── README.md                         # 📄 Documentación
```

## 🚀 Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd proyecto_track_people
   ```

2. **Crear un entorno virtual (Recomendado)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎥 Uso

### 1. Detección en Tiempo Real
Para ejecutar el sistema de seguimiento utilizando tu webcam (o un video):

```bash
python realtime.py
```
*   **Controles**: Presiona `q` para salir de la ventana.
*   **Nota**: Si usas un archivo de video en lugar de webcam, edita la variable `video_source` en `realtime.py`.

### 2. Entrenamiento del Modelo
Si deseas re-entrenar el modelo con nuevos datos:
1.  Abre el notebook `notebooks/entrenamiento_yolo.ipynb`.
2.  Asegúrate de tener tu dataset configurado en formato YOLO.
3.  Ejecuta las celdas para iniciar el entrenamiento.

## 📊 Rendimiento del Modelo

El modelo actual (`overhead_people_tracking.pt`) obtuvo las siguientes métricas tras 50 épocas:

| Métrica | Valor | Descripción |
| :--- | :--- | :--- |
| **mAP@50** | **98.3%** | Precisión media con IoU de 0.5. |
| **mAP@50-95** | **70.7%** | Precisión robusta en diferentes umbrales. |
| **Precision** | **97.0%** | Tasa de verdaderos positivos. |
| **Recall** | **96.7%** | Capacidad para encontrar todas las personas. |

## 🛠️ Tecnologías

*   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Detección de objetos SOTA.
*   OpenCV - Procesamiento de imágenes en tiempo real.
*   Pandas & Matplotlib - Análisis y visualización de métricas.

---
Desarrollado por Hugo.
