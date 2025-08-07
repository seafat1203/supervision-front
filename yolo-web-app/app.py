import os
import uuid
import cv2
from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
import supervision as sv

# 初始化 Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "tmp"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB 文件限制
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 加载 YOLO 模型
model = YOLO("yolov8n.pt")  # 可替换为 yolov8s.pt / yolov8m.pt 等

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

    uid = str(uuid.uuid4())
    input_filename = f"{uid}.jpg"
    output_filename = f"{uid}-output.jpg"
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

    file.save(input_path)

    # 检查图片尺寸
    image = cv2.imread(input_path)
    if image is None:
        return "❌ 无法读取上传的图片", 400

    h, w = image.shape[:2]
    if h > 3000 or w > 3000:
        os.remove(input_path)
        return "❌ 图片尺寸太大（最大支持 3000x3000）", 400

    # 目标检测
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

@app.errorhandler(413)
def request_entity_too_large(error):
    return "❌ 上传的图片太大啦（最大 8MB）", 413

# ✅ Render 要求绑定到 PORT 环境变量
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port)