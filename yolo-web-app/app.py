import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
import supervision as sv
from collections import Counter

# ====== 基础路径 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_TMP_DIR = os.path.join(BASE_DIR, "static", "tmp")
os.makedirs(STATIC_TMP_DIR, exist_ok=True)

# ====== Flask 初始化 ======
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = STATIC_TMP_DIR
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB

# ====== 加载模型 ======
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
print("🚀 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded")

# ====== 首页 ======
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ====== 健康检查 ======
@app.route("/health")
def health():
    return {"status": "ok"}


# ====== 目标检测 ======
@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return "❌ 没有找到上传文件", 400

    file = request.files["image"]

    if file.filename == "":
        return "❌ 文件名为空", 400

    uid = str(uuid.uuid4())

    input_filename = f"{uid}.jpg"
    output_filename = f"{uid}-output.jpg"

    input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

    file.save(input_path)

    # ====== 读取图片 ======
    image = cv2.imread(input_path)

    if image is None:
        return "❌ 无法读取上传的图片", 400

    # 轻微增强
    image = cv2.convertScaleAbs(image, alpha=1.1, beta=20)

    # ====== YOLO推理 ======
    results = model(image, conf=0.2, iou=0.6)[0]

    detections = sv.Detections.from_ultralytics(results)

    # ====== 画框 ======
    box_annotator = sv.BoxAnnotator(thickness=2)

    label_annotator = sv.LabelAnnotator(
        text_scale=0.6,
        text_thickness=1
    )

    labels = [
        f"{model.model.names[cid]} {score:.2f}"
        for cid, score in zip(
            detections.class_id,
            detections.confidence
        )
    ]

    annotated = box_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    success = cv2.imwrite(output_path, annotated)

    print(f"✅ 图片保存状态: {success}, 路径为: {output_path}")

    # ====== 统计识别结果 ======
    cls_ids = [int(c) for c in detections.class_id]

    counts = Counter(cls_ids)

    summary_parts = [
        f"{model.model.names[k]} × {v}"
        for k, v in counts.items()
    ]

    summary_text = (
        "，".join(summary_parts)
        if summary_parts
        else "未识别到目标"
    )

    return render_template(
        "index.html",
        result_img=f"/static/tmp/{output_filename}",
        result_summary=summary_text
    )


# ====== 静态图片访问 ======
@app.route("/static/tmp/<filename>")
def result_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ====== 错误处理 ======
@app.errorhandler(413)
def request_entity_too_large(error):
    return "❌ 上传的图片太大啦（最大 8MB）", 413


# ====== 启动 ======
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    print("===================================")
    print("🚀 AI Image Detection Server Start")
    print(f"📡 Port: {port}")
    print("===================================")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )