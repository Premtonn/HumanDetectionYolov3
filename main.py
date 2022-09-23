import cv2
import numpy as np
import subprocess

from flask import Flask, render_template, Response

app = Flask(__name__)



net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')


classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

width = 1920
height = 1080

# cap = cv2.VideoCapture("rtmp://10.212.26.179:1935/live")
cap = cv2.VideoCapture(0)
x = cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
y = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
enable_gpu = True

skip_frames = False






def gen_frames():
    while True:
        _, img = cap.read()

        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    if class_id == 0:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)
                        # print(len(class_ids))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        count = 0
        countText = ""

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])

                if label == "person":
                    color = colors[i]

                    confidence = str(round(confidences[i], 2))
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    count += 1
                    countText = "Count: " + str(count)

                    cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
        cv2.putText(img, countText, (7, 50), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', img)
        frameToShow = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frameToShow + b'\r\n')  # concat frame one by one and show result

        # print(class_ids)
        # print(count)

        # img = cv2.resize(img, (1280, 780))
        cv2.imshow('Transmitting Feed', img)
        key = cv2.waitKey(1)
        if key == 27:
            break

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

cap.release()
cv2.destroyAllWindows()

