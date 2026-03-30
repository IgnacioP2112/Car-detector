# Detector de Vehículos con YOLOv8

Sistema de detección de vehículos en video utilizando YOLOv8 y OpenCV.
El programa procesa un video frame a frame, identifica vehículos, muestra resultados en tiempo real y entrega estadísticas al finalizar.

---

## Descripción

Este proyecto implementa un pipeline de visión computacional para detectar vehículos en video. Inicialmente se consideró el uso de Haar Cascade, pero se optó por YOLOv8 debido a su superior precisión en escenarios urbanos complejos.

El sistema realiza las siguientes tareas:

- Detecta vehículos (autos, motos, buses y camiones).
- Dibuja cuadros delimitadores (bounding boxes) con etiquetas y nivel de confianza.
- Muestra el número de vehículos detectados por cada frame.
- Calcula los FPS (cuadros por segundo) en tiempo real.
- Genera un resumen estadístico al finalizar la ejecución.

---

## Tecnologías Utilizadas

- Python
- OpenCV
- YOLOv8 (Ultralytics)

---

## Estructura del Proyecto

```text
car_detector/
├── main.py          # Orquestación del programa
├── detector.py      # Lógica de detección
├── config.py        # Parámetros configurables
├── assets/          # Archivos auxiliares (referencia)
├── requirements.txt
└── .gitignore
```

---

## Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/IgnacioP2112/Car-detector.git
cd Car-detector
```

2. Crear un entorno virtual (recomendado):

```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

3. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

---

## Uso

1. Agregar un video de tráfico en la carpeta `video/`.
2. Configurar la ruta de origen en el archivo `config.py`:
   `VIDEO_SOURCE = "video/video.mp4"`
3. Ejecutar el script principal:
   ```bash
   python main.py
   ```
4. Presionar la tecla **q** para salir de la visualización.

---

## Resultados

### Visualización en tiempo real
El sistema despliega una ventana que muestra:
- Detecciones dibujadas sobre el video.
- Conteo de vehículos presentes en el frame actual.
- Rendimiento del sistema (FPS).

### Resumen estadístico
Al cerrar el programa, se imprime en la consola:
- Total de frames procesados.
- Promedio de vehículos por frame.
- Valores máximos y mínimos de detección.
- Total acumulado de detecciones procesadas.

---

## Consideraciones Técnicas

- El conteo se realiza por cada frame individual, no por vehículo único.
- Esta versión no implementa tracking (seguimiento) de objetos.
- El rendimiento de procesamiento depende directamente del hardware y del modelo seleccionado (por defecto `yolov8n.pt`).

---

## Trabajo Futuro

- Implementar algoritmos de tracking (como SORT o DeepSORT) para el conteo de vehículos únicos.
- Clasificar y separar el conteo por categorías específicas (ej. solo motos o solo camiones).
- Evaluar el impacto en la precisión utilizando modelos más robustos como `yolov8s.pt` o `yolov8m.pt`.