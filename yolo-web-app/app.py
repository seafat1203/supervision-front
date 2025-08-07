from flask import Flask, request, send_file
import os
import uuid
import cv2
from ultralytics import YOLO
import supervision as sv

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # 可换成 s/m 等更强模型

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    if not file:
        return {"error": "没有上传文件"}, 400

    uid = str(uuid.uuid4())
    input_path = f"tmp/{uid}.jpg"
    output_path = f"tmp/{uid}-output.jpg"

    file.save(input_path)

    # 推理
    image = cv2.imread(input_path)
    results = model(image, conf=0.3)[0]
    detections = sv.Detections.from_ultralytics(results)

    # 可视化
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)
    box_annotator = sv.BoxAnnotator(thickness=2)
    labels = [
        f"{model.model.names[cid]} {score:.2f}"
        for cid, score in zip(detections.class_id, detections.confidence)
    ]
    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    cv2.imwrite(output_path, annotated)

    return send_file(output_path, mimetype='image/jpeg')