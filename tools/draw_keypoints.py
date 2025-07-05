import os
import cv2
import numpy as np
import onnxruntime as ort

input_size = (256, 192)
mean = np.array([123.675, 116.28, 103.53]) / 255.0
std = np.array([58.395, 57.12, 57.375]) / 255.0
simcc_split = 2.0

CONF_THRESH = 0.25
NMS_THRESH = 0.45

# Updated POSE_PAIRS for correct connection
POSE_PAIRS = [
    [5,6], [6, 8], [8,10], [6,12],      # nose → right_eye → right_ear
    [12,14], [14,16], [5, 7],               # nose → left_eye → left_ear
    [7,9], [5,11], [11, 13],          # left_shoulder → left_hip → left_knee → left_ankle
    [13,15]                     # shoulders and hips connected
]

def preprocess_yolo(image):
    h, w = image.shape[:2]
    scale = 640.0 / max(h, w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    pad_img = np.zeros((640, 640, 3), dtype=np.uint8)
    pad_img[:resized.shape[0], :resized.shape[1]] = resized
    img = pad_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0), scale

def postprocess_yolo(output, scale, conf_thresh=CONF_THRESH):
    boxes = output[0].squeeze(0)
    boxes = boxes[boxes[:, 4] > conf_thresh]
    if boxes.shape[0] == 0:
        return []
    scores = boxes[:, 4:5] * boxes[:, 5:]
    labels = np.argmax(scores, axis=1)
    confs = np.max(scores, axis=1)
    keep = (labels == 0) & (confs > conf_thresh)
    boxes = boxes[keep]
    confs = confs[keep]
    labels = labels[keep]

    if boxes.shape[0] > 1:
        top = np.argmax(confs)
        boxes = boxes[top:top+1]
        confs = confs[top:top+1]
        labels = labels[top:top+1]

    xywh = boxes[:, :4]
    cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = (cx - w / 2) / scale
    y1 = (cy - h / 2) / scale
    x2 = (cx + w / 2) / scale
    y2 = (cy + h / 2) / scale
    bboxes = np.stack([x1, y1, x2, y2], axis=1)
    results = []
    for i in range(bboxes.shape[0]):
        results.append(np.concatenate([bboxes[i], [confs[i]], [labels[i]]]))
    return np.array(results)

def filter_person_detections(dets, conf_thresh=0.4):
    dets = np.array(dets)
    if len(dets.shape) != 2 or dets.shape[1] < 6:
        return []
    person_dets = []
    for d in dets:
        x1, y1, x2, y2, score, cls_id = d
        if int(cls_id) == 0 and score > conf_thresh:
            person_dets.append([x1, y1, x2, y2])
    return np.array(person_dets, dtype=np.int32)

def decode_simcc(simcc_x, simcc_y, bbox_width, bbox_height):
    """解码 SIMCC 格式的关键点，并映射到 bbox 尺寸坐标系"""
    coords = []
    num_kpts, simcc_dim_x = simcc_x.shape
    _, simcc_dim_y = simcc_y.shape

    scale_x = bbox_width / simcc_dim_x
    scale_y = bbox_height / simcc_dim_y

    for i in range(num_kpts):
        x_idx = np.argmax(simcc_x[i])
        y_idx = np.argmax(simcc_y[i])
        score = (simcc_x[i][x_idx] + simcc_y[i][y_idx]) / 2
        coord_x = x_idx * scale_x
        coord_y = y_idx * scale_y
        coords.append([coord_x, coord_y])
    return np.array(coords)

def draw_keypoints(image, keypoints, offset, radius=3):
    ox, oy = offset
    for idx, (x, y) in enumerate(keypoints):
        px = int(x + ox)
        py = int(y + oy)
        cv2.circle(image, (px, py), radius, (0, 255, 0), -1)
        cv2.putText(image, str(idx), (px + 2, py - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

    for pair in POSE_PAIRS:
        p1, p2 = pair
        if p1 < len(keypoints) and p2 < len(keypoints):
            x1, y1 = keypoints[p1]
            x2, y2 = keypoints[p2]
            pt1 = (int(x1 + ox), int(y1 + oy))
            pt2 = (int(x2 + ox), int(y2 + oy))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)
    return image

def preprocess_pose(crop_img):
    if crop_img is None or crop_img.size == 0:
        raise ValueError("Empty crop image for pose estimation")
    img = cv2.resize(crop_img, input_size[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
    return img

def draw_bbox(image, bbox, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def crop_and_infer_pose(pose_session, image, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    crop_img = image[y1:y2, x1:x2]
    if crop_img is None or crop_img.size == 0:
        raise ValueError(f"Invalid crop with bbox: {bbox}")
    input_tensor = preprocess_pose(crop_img)
    input_name = pose_session.get_inputs()[0].name
    simcc_x, simcc_y = pose_session.run(None, {input_name: input_tensor})
    keypoints = decode_simcc(simcc_x[0], simcc_y[0], bbox_width, bbox_height)
    return keypoints, (x1, y1)

def infer_yolo(yolo_session, image):
    input_tensor, scale_factors = preprocess_yolo(image)
    input_name = yolo_session.get_inputs()[0].name
    outputs = yolo_session.run(None, {input_name: input_tensor})
    dets = postprocess_yolo(outputs, scale_factors)
    return dets

def main():
    image_path = '/home/ljy/project/tensorrt_yolov5/data/pictures/000000000785.jpg'
    yolo_onnx_path = '/home/ljy/project/poseDetection/models/onnx/yolov5s.onnx'
    pose_onnx_path = '/home/ljy/project/poseDetection/models/rtmpose_model/rtmpose.onnx'

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    yolo_session = ort.InferenceSession(yolo_onnx_path, providers=['CPUExecutionProvider'])
    pose_session = ort.InferenceSession(pose_onnx_path, providers=['CPUExecutionProvider'])

    dets = infer_yolo(yolo_session, image)
    person_bboxes = filter_person_detections(dets)

    if not person_bboxes.any():
        print("[Warning] No person detections above threshold.")

    for bbox in person_bboxes:
        draw_bbox(image, bbox)
        try:
            keypoints, offset = crop_and_infer_pose(pose_session, image, bbox)
            draw_keypoints(image, keypoints, offset)
        except Exception as e:
            print(f"[Error] Skipping bbox {bbox} due to: {e}")

    save_path = os.path.join(os.getcwd(), 'output_pose.jpg')
    cv2.imwrite(save_path, image)
    print(f"Saved visualization to: {save_path}")

if __name__ == '__main__':
    main()
