from ultralytics import YOLO
import cv2
import argparse
import os


DEFAULT_MODEL = 'v8nano2.pt'


class DroniadaDetector:
    def __init__(self, conf=0.5, iou=0.5, model=DEFAULT_MODEL):
        model_path = os.path.join(os.path.dirname(__file__), 'models', model)
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def detect(self, image, conf=None, iou=None, visualize=False):
        conf = conf or self.conf
        iou = iou or self.iou
        results = self.model(image, conf=conf, iou=iou, verbose=False)[0]
        boxes = results.boxes
        xywh = [box.xywh.numpy() for box in boxes]
        if visualize:
            annotated_image = results.plot()
            cv2.imshow('Droniada Detector', annotated_image)
            cv2.waitKey(1)
        return xywh


def detect_camera():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='0')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--iou', type=float, default=0.5)
    args = parser.parse_args()

    detector = DroniadaDetector(conf=args.conf, iou=args.iou)
    cap = cv2.VideoCapture(int(args.src) if args.src.isdigit() else args.src)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detector.detect(frame, visualize=True)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_camera()
