from ultralytics import YOLO
import torch
import cv2

# 加载模型
#model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt", task='detect')  # 加载预训练模型（建议用于训练）
# 使用模型
#model.train(data="coco128.yaml", epochs=3)  # 训练模型
#metrics = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
print(results[0])
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)

#success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
