import os
import uuid
import cv2
from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
import supervision as sv

# 初始化 Flask 应用
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "tmp"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 限制上传大小为8MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 加载 YOLO 模型
model = YOLO("yolov8n.pt")  # 可替换为 yolov8s.pt 等

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return "❌ 没有找到上传文件", 400

    file = request.files["image"]
    if file.filename == "":
        return "❌ 文件名为空", 400

    # 保存原图
    uid = str(uuid.uuid4())
    input_filename = f"{uid}.jpg"
    output_filename = f"{uid}-output.jpg"
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
    file.save(input_path)

    # 读取图片并检查尺寸
    image = cv2.imread(input_path)
    h, w = image.shape[:2]
    if h > 3000 or w > 3000:
        os.remove(input_path)
        return "❌ 图片尺寸太大（最大支持3000x3000）", 400

    # 推理
    results = model(image, conf=0.3)[0]
    detections = sv.Detections.from_ultralytics(results)

    # 可视化
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)
    labels = [
        f"{model.model.names[cid]} {score:.2f}"
        for cid, score in zip(detections.class_id, detections.confidence)
    ]
    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    cv2.imwrite(output_path, annotated)

    return render_template("index.html", result_img=f"/result/{output_filename}")

@app.route("/result/<filename>")
def result_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# 错误处理：文件过大
@app.errorhandler(413)
def request_entity_too_large(error):
    return "❌ 上传的图片太大啦（最大8MB）", 413