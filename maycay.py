import os
import sys
import time
import argparse
from datetime import datetime

# Xóa biến QT_QPA_PLATFORM nếu bị set nhầm plugin offscreen không tồn tại
os.environ.pop('QT_QPA_PLATFORM', None)

# Chỉ bật GUI khi có cờ ENABLE_GUI=1 và có DISPLAY hợp lệ
DEBUG_SAVE = False
SHOW_GUI = (os.environ.get('ENABLE_GUI', '').lower() in ['1', 'true', 'yes']) and bool(os.environ.get('DISPLAY'))

# Tim client_lib du o /workspace hay thu muc hien tai
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="UIT Car Racing - Autonomous Driving")
parser.add_argument("--collect", type=str, default="", help="Ten thu muc luu anh data (vd: map1)")
parser.add_argument("--model", type=str, default="", help="Duong dan den model yolo (vd: map1_best.pt)")
args, unknown = parser.parse_known_args()

COLLECT_DIR = None
if args.collect:
    COLLECT_DIR = os.path.join(CURRENT_DIR, "dataset", "raw", args.collect)
    os.makedirs(COLLECT_DIR, exist_ok=True)
    print(f"[INFO] CHE DO THU THAP DU LIEU: Anh se duoc luu vao {COLLECT_DIR}", flush=True)

sys.path.extend(["/workspace", CURRENT_DIR])

from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Tu dong tim duong dan trong so best.pt linh hoat
MODEL_PATHS = [
    "/workspace/best.pt",
    os.path.join(CURRENT_DIR, "Road_Seg_Model", "modelYolo", "weights", "best.pt"),
    "/workspace/UIT-CAR-RACING/Road_Seg_Model/modelYolo/weights/best.pt",
    "/workspace/Road_Seg_Model/modelYolo/weights/best.pt",
    "Road_Seg_Model/modelYolo/weights/best.pt"
]
if args.model:
    model_path = args.model
    if not os.path.exists(model_path) and os.path.exists(os.path.join(CURRENT_DIR, model_path)):
        model_path = os.path.join(CURRENT_DIR, model_path)
else:
    model_path = next((p for p in MODEL_PATHS if os.path.exists(p)), MODEL_PATHS[0])
model = YOLO(model_path)
print(f"[INFO] Load model: {model_path}", flush=True)

class LastSpeed:
    value = 30

def get_yolo_segmentation(raw_image):
    results = model.predict(source=raw_image, verbose=False)

    if not results or results[0].masks is None:
        return np.zeros(raw_image.shape[:2], dtype=np.uint8)

    masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
    combined_mask = (np.sum(masks, axis=0) > 0).astype(np.uint8) * 255
    return combined_mask

class LaneWidthEstimator:
    def __init__(self):
        self.base_width = None
        self.current_width = None
        self.width_profile = []  # luu tat ca phep do (y, width)

    def measure_width(self, gray_image, green_line_points, step=5, fraction=2):
        """Do do rong duong theo nhieu dong y (tu day anh len 1/fraction chieu cao)"""
        height, width = gray_image.shape
        max_y = height - height // 3
        min_y = height - height // fraction

        widths = []
        self.width_profile = []

        for y in range(max_y, min_y, -step):
            row = gray_image[y, :]
            non_zero_cols = np.where(row > 0)[0]
            if len(non_zero_cols) > 1:
                left = non_zero_cols[0]
                right = non_zero_cols[-1]
                lane_width = right - left
                widths.append(lane_width)
                self.width_profile.append((y, lane_width))

        if widths:
            self.current_width = np.median(widths)
            if self.base_width is None:
                self.base_width = self.current_width

        return self.current_width, self.width_profile

    def is_wide_change(self, threshold=0.2):
        """Chi can 1 phep do CurW > base_width * (1+threshold) la coi nhu duong rong"""
        if self.base_width is None or not self.width_profile:
            return False
        for _, w in self.width_profile:
            if w > self.base_width * (1 + threshold):
                return True
        return False


def show_image_safe(win_name, img):
    """Hien thi anh an toan neu co moi truong do hoa hoac luu anh debug"""
    if SHOW_GUI:
        try:
            # Thu nhỏ ảnh 50% để XLaunch không bị lag làm tụt FPS
            small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow(win_name, small_img)
        except Exception:
            pass
            
    if DEBUG_SAVE:
        debug_dir = os.path.join(CURRENT_DIR, "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)
        # Chỉ lưu đè 1 file để dễ xem live, hoặc lưu vài file
        cv2.imwrite(os.path.join(debug_dir, f"{win_name}_live.jpg"), img)


def calculate_steering_angle(segment_image, speed=30,
                             k_min=0.06, k_max=0.3, max_speed=35,
                             lane_width_estimator=None):
    if len(segment_image.shape) == 3:
        gray = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = segment_image.copy()

    height, width = gray.shape
    green_line_points = []

    for i in range(height - 1, -1, -1):
        row = gray[i, :]
        non_zero_cols = np.where(row > 0)[0]
        if len(non_zero_cols) > 0:
            cx_row = int(np.mean(non_zero_cols))
            green_line_points.append((cx_row, i))

    lane_cx = width // 2
    curve_error = 0
    near_error = 0
    far_error = 0

    near_points = [(cx, y) for cx, y in green_line_points if y >= height // 2]
    far_points = [(cx, y) for cx, y in green_line_points if y < height // 3]

    wide_change = False
    if lane_width_estimator is not None:
        cur_w, width_profile = lane_width_estimator.measure_width(gray, green_line_points)
        wide_change = lane_width_estimator.is_wide_change()

    if wide_change:
        far_points = []

    if green_line_points:
        weighted_sum = sum(cx * (height - y) for cx, y in green_line_points)
        total_weight = sum(height - y for _, y in green_line_points)
        lane_cx = int(weighted_sum / total_weight)

        bottom_row_cx = green_line_points[-1][0]
        top_row_cx = green_line_points[0][0]
        curve_error = top_row_cx - bottom_row_cx

        near_error = int(np.mean([cx for cx, y in near_points])) - width // 2 if near_points else 0
        far_error = int(np.mean([cx for cx, y in far_points])) - width // 2 if far_points else 0

    abs_error = abs(near_error) + abs(far_error)
    if abs_error < 10:
        w_near = 0.7
        w_far = 0.3
    else:
        w_far = 0.5 + 0.25 * (speed / max_speed)
        w_far = np.clip(w_far, 0.35, 0.75)
        w_near = 1 - w_far - 0.05
        w_near = np.clip(w_near, 0.25, 0.65)
        if near_error * far_error < 0:
            w_far *= 0.7
            w_near = 1 - w_far - 0.05
            w_near = np.clip(w_near, 0.25, 0.65)

    blended_error = w_near * near_error + w_far * far_error
    if abs(blended_error) < 3:
        blended_error = 0

    error_ratio = min(abs(blended_error) / (width // 2), 1.0)
    curve_ratio = min(abs(curve_error) / (width // 2), 1.0)

    k = k_min + (k_max - k_min) * (curve_ratio ** 0.5)
    k *= (1 + 1.0 * error_ratio)
    base_angle = blended_error * k

    speed_factor = 1.0 + 1.2 * (speed / max_speed) * (1 - 0.5 * (error_ratio ** 2))
    aggressive_factor = 1.0 + 5.0 * (error_ratio ** 2) if error_ratio > 0.1 else 1.0
    normal_angle = base_angle * speed_factor * aggressive_factor
    normal_angle = np.clip(normal_angle, -15, 15)

    if blended_error >= 35:
        angle = 25
    elif blended_error <= -35:
        angle = -25
    elif 25 <= blended_error < 35:
        scale = (blended_error - 25) / 20
        angle = 18 + 12 * scale
    elif -35 < blended_error <= -25:
        scale = (-blended_error - 25) / 20
        angle = -(18 + 12 * scale)
    elif 15 <= blended_error < 25:
        scale = (blended_error - 20) / 20
        angle = 10 + 10 * scale
    elif -25 < blended_error <= -15:
        scale = (-blended_error - 20) / 20
        angle = -(10 + 10 * scale)
    else:
        angle = normal_angle

    angle = np.clip(angle, -25, 25)

    debug = np.zeros((height, width, 3), dtype=np.uint8)
    debug[gray > 0] = (90, 90, 90)
    for cx, y in green_line_points:
        debug[y, cx] = (0, 255, 0)

    cv2.line(debug, (lane_cx, 0), (lane_cx, height), (0, 0, 255), 2)

    if far_points:
        far_pts = np.array([[cx, y] for cx, y in far_points], np.int32)
        far_pts = far_pts.reshape((-1, 1, 2))
        cv2.polylines(debug, [far_pts], isClosed=False, color=(255, 0, 0), thickness=2)

    if near_points:
        near_cx = int(np.mean([cx for cx, y in near_points]))
        cv2.line(debug, (near_cx, height // 2), (near_cx, height), (0, 255, 255), 2)

    if lane_width_estimator and lane_width_estimator.current_width:
        for (y_ref, w_ref) in lane_width_estimator.width_profile:
            cx_ref = lane_cx
            half_w = int(w_ref // 2)
            left_x = max(0, cx_ref - half_w)
            right_x = min(width - 1, cx_ref + half_w)
            cv2.line(debug, (left_x, y_ref), (right_x, y_ref), (0, 200, 255), 1)

        base_w = lane_width_estimator.base_width or 0
        cur_w = lane_width_estimator.current_width
        cv2.putText(debug, f"BaseW: {base_w:.1f} CurW(median): {cur_w:.1f}",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if wide_change:
            cv2.putText(debug, "WIDE ROAD DETECTED - Trust NEAR",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    show_image_safe("Lane Debug", debug)
    return angle, blended_error, k, w_near, w_far, curve_ratio


if __name__ == "__main__":
    print("[INFO] Da nap xong YOLO Model! Dang khoi tao bo dieu khien...")
    angle_history = deque(maxlen=2)
    last_angle = 0
    max_delta = 15

    max_speed = 35
    min_speed = 25
    last_speed = LastSpeed()

    lane_width_estimator = LaneWidthEstimator()
    print("[INFO] Bat dau vong lap nhan dien & dieu khien (Dang ket noi Unity)...", flush=True)
    print("  -> Hay dam bao Game Unity dang mo va da vao sa hinh!", flush=True)

    frame_count = 0
    waiting_count = 0
    connected_announced = False

    try:
        while True:
            try:
                state = GetStatus()
                raw_image = GetRaw()
            except Exception as e:
                time.sleep(0.05)
                continue

            if raw_image is None or raw_image.size == 0:
                waiting_count += 1
                if waiting_count % 30 == 0:
                    print("[INFO] Dang cho frame anh tu Unity... (Neu chua bat game, hay khoi dong Unity)", flush=True)
                time.sleep(0.05)
                continue

            if not connected_announced:
                print(f"[INFO] >>> DA KET NOI THANH CONG! Nhan frame {raw_image.shape} tu Unity <<<", flush=True)
                connected_announced = True

            waiting_count = 0
            frame_count += 1
            
            if COLLECT_DIR and frame_count % 3 == 0:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{args.collect}_{timestamp}.jpg"
                    cv2.imwrite(os.path.join(COLLECT_DIR, filename), raw_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                except Exception:
                    pass
                    
            seg_mask = get_yolo_segmentation(raw_image)

            # Luu anh truc quan hoa (Tam thoi tat de toi uu FPS)
            # if frame_count % 5 == 0:
            #     try:
            #         colored_mask = cv2.applyColorMap(seg_mask, cv2.COLORMAP_JET)
            #         blend = cv2.addWeighted(raw_image, 0.6, colored_mask, 0.4, 0)
            #         combined_view = np.hstack((raw_image, blend))
            #         cv2.imwrite(os.path.join(CURRENT_DIR, "live_view.jpg"), combined_view)
            #     except Exception:
            #         pass

            show_image_safe('Raw Image', raw_image)
            show_image_safe("YOLO Segmentation", seg_mask)

            angle, blended_error, k, w_near, w_far, curve_ratio = calculate_steering_angle(
                seg_mask,
                max_speed=max_speed,
                speed=last_speed.value,
                lane_width_estimator=lane_width_estimator
            )

            speed = max(min_speed, max_speed * (1 - 0.7 * curve_ratio))
            speed = 0.6 * last_speed.value + 0.4 * speed
            last_speed.value = speed

            if abs(blended_error) < 20:
                angle_history.append(angle)
            else:
                angle_history.clear()

            if angle_history:
                smoothed_angle = sum(angle_history) / len(angle_history)
                smoothed_angle = max(min(smoothed_angle, last_angle + max_delta),
                                     last_angle - max_delta)
            else:
                smoothed_angle = angle
            last_angle = smoothed_angle

            try:
                AVControl(float(speed), float(smoothed_angle))
            except Exception as e:
                if frame_count % 10 == 0:
                    print(f"[ERROR AVControl]: {e}", flush=True)

            if frame_count % 3 == 0 or frame_count == 1:
                print(f"[Frame {frame_count:04d}] Speed: {speed:.1f} | Angle: {smoothed_angle:+.2f} | Error: {blended_error:+.1f}", flush=True)

            if SHOW_GUI:
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

    finally:
        print('Closing socket and windows...')
        try:
            CloseSocket()
            cv2.destroyAllWindows()
        except Exception:
            pass
