# droniada-detect

## Install

```bash
pip install git+https://github.com/AGH-Drone-Engineering/droniada-detect
```

## Run

```bash
droniada_detector
```

Arguments:

* `--src` - camera index or video file
* `--conf` - confidence threshold
* `--iou` - IoU threshold

## Use as a library

```python
from droniada_detect import DroniadaDetector
detector = DroniadaDetector()
img = cv2.imread('test.jpg')
boxes = detector.detect(img)
print(boxes)
# [[x, y, w, h, "white_square|brown_circle|white_circle|gold_circle"], ...]
```
