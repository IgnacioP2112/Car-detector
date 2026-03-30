import cv2
import time
from detector import CarDetector
from config import VIDEO_SOURCE, VEHICLE_CLASSES, CONFIDENCE_THRESHOLD


def mostrar_resumen(conteos):
    if not conteos:
        print("No se procesaron frames.")
        return

    print("\n=== Resumen del análisis ===")
    print(f"Frames procesados:              {len(conteos)}")
    print(f"Promedio de vehículos por frame: {sum(conteos) / len(conteos):.1f}")
    print(f"Máximo detectado en un frame:   {max(conteos)}")
    print(f"Mínimo detectado en un frame:   {min(conteos)}")
    print(f"Total de detecciones:           {sum(conteos)}")
    print("============================")


def main():
    detector = CarDetector(VEHICLE_CLASSES, CONFIDENCE_THRESHOLD)
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"Error: no se pudo abrir el video: {VIDEO_SOURCE}")
        return

    print("Procesando video... presiona 'q' para salir.")

    conteos = []
    frame_numero = 0
    loop_count = 0

    #Variables para FPS
    start_time = time.time()
    frames_procesados = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            loop_count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        detections = detector.detect(frame)
        frame_with_boxes = detector.draw_detections(frame, detections)

        count = len(detections)

        if loop_count == 0:
            conteos.append(count)
            frame_numero += 1
            frames_procesados += 1  #  contar frames para FPS

        #Cálculo de FPS
        elapsed_time = time.time() - start_time
        fps = frames_procesados / elapsed_time if elapsed_time > 0 else 0

        # ====== CONTADOR DE VEHÍCULOS ======
        texto = f"Vehiculos: {count}"
        (ancho_texto, alto_texto), _ = cv2.getTextSize(
            texto, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(frame_with_boxes, (10, 10), (20 + ancho_texto, 45), (0, 0, 0), -1)
        cv2.putText(
            frame_with_boxes,
            texto,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # ====== ESTADO DEL VIDEO ======
        if loop_count == 0:
            estado = f"Frame: {frame_numero}"
        else:
            estado = "Repeticion del video"

        cv2.rectangle(frame_with_boxes, (10, 50), (280, 80), (0, 0, 0), -1)
        cv2.putText(
            frame_with_boxes,
            estado,
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        # ====== FPS ======
        texto_fps = f"FPS: {fps:.2f}"
        cv2.rectangle(frame_with_boxes, (10, 85), (150, 115), (0, 0, 0), -1)
        cv2.putText(
            frame_with_boxes,
            texto_fps,
            (15, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Detector de Vehiculos - YOLOv8", frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mostrar_resumen(conteos)


if __name__ == "__main__":
    main()