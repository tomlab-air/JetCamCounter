#!/usr/bin/env python3
import os
import time
import argparse
import cv2
import numpy as np
from datetime import datetime

# --- Settings ---------------------------------------------------
CONF_THRESHOLD = 0.3      # Detection confidence threshold
NMS_THRESHOLD  = 0.45     # Non-max suppression threshold
INPUT_W, INPUT_H = 416, 416

CFG_PATH     = "cfg/yolov4-tiny.cfg"
DATA_PATH    = "cfg/coco.data"
WEIGHTS_PATH = "yolov4-tiny.weights"
# ---------------------------------------------------------------

class JetCamCounter:
    """
    JetCamCounter encapsulates all logic for:
    - Opening a video source (camera or file)
    - Running YOLOv4-tiny detection via Darknet
    - Tracking and counting objects crossing a center line
    - Saving annotated video output and a timestamped count log
    """

    def __init__(self, input_source: str = None):
        """
        Initialize instance variables, determine mode (camera vs video), 
        set up output filenames (with timestamp), and open video capture.
        """
        # Determine mode and base name for output
        if input_source:
            self.source_str = input_source
            self.base_name = os.path.splitext(os.path.basename(input_source))[0]
            self.mode = "video"
            print(f"Processing video file '{self.source_str}' as input source.")
        else:
            self.source_str = None
            self.base_name = "camera"
            self.mode = "camera"
            print("Processing camera (0) as input source.")

        # Timestamp suffix for unique filenames
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure result directory exists
        os.makedirs("result", exist_ok=True)

        # Compose output filenames
        self.output_video_name = f"{self.base_name}_{self.mode}_{self.run_timestamp}.mp4"
        self.log_filename      = f"{self.base_name}_{self.mode}_count_log_{self.run_timestamp}.txt"
        self.output_path = os.path.join("result", self.output_video_name)
        self.log_path    = os.path.join("result", self.log_filename)

        # Tracking & counting state
        self.next_id    = 0
        self.tracked    = {}     # { id: {'last_x', 'counted'} }
        self.total_cnt  = 0
        self.class_cnts = {}     # {'car': 2, ...}

        # Prepare video capture
        self.cap = self._open_capture()

        # After opening capture, initialize Darknet network
        import darknet
        self.darknet = darknet
        self.network, self.class_names, self.class_colors = self.darknet.load_network(
            CFG_PATH, DATA_PATH, WEIGHTS_PATH, batch_size=1
        )

        # Retrieve original frame dimensions and FPS
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30

        # Compute scaling factors for mapping between YOLO input and original resolution
        self.scale_x  = self.orig_w / INPUT_W
        self.scale_y  = self.orig_h / INPUT_H
        self.center_x = self.orig_w // 2

        # Prepare video writer
        self.out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps, (self.orig_w, self.orig_h)
        )

        # Create and resize display window
        cv2.namedWindow("JetCamCounter", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("JetCamCounter", self.orig_w, self.orig_h)

    def _open_capture(self):
        """
        Open cv2.VideoCapture using camera (0) or video file path.
        Raise if it fails.
        """
        if self.source_str:
            cap = cv2.VideoCapture(self.source_str)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Could not open input source")
        return cap

    def _detect_and_count(self, frame, log_file):
        """
        Perform YOLO detection on the given frame, do simple tracking by
        comparing horizontal distances, count objects that cross the center line,
        and log timestamp + class name whenever a new crossing is detected.
        Return the annotated frame and updated tracking state.
        """
        # 1) Resize to YOLO input size and convert to RGB
        rgb_img = cv2.cvtColor(cv2.resize(frame, (INPUT_W, INPUT_H)), cv2.COLOR_BGR2RGB)
        img_for_dark = self.darknet.make_image(INPUT_W, INPUT_H, 3)
        self.darknet.copy_image_from_bytes(img_for_dark, rgb_img.tobytes())
        detections = self.darknet.detect_image(
            self.network, self.class_names, img_for_dark,
            thresh=CONF_THRESHOLD, hier_thresh=0.5,
            nms=NMS_THRESHOLD
        )
        self.darknet.free_image(img_for_dark)

        current = {}
        for label, conf, (x, y, bw, bh) in detections:
            # Map detection from resized frame back to original resolution
            cx = x * self.scale_x
            cy = y * self.scale_y
            w0 = bw * self.scale_x
            h0 = bh * self.scale_y
            x1 = int(cx - w0 / 2)
            y1 = int(cy - h0 / 2)
            cx = int(cx)

            # Match with previously tracked objects by nearest horizontal distance
            assigned = None
            best_dist = float("inf")
            for oid, info in self.tracked.items():
                dist = abs(info['last_x'] - cx)
                if dist < best_dist and dist < w0:
                    best_dist = dist
                    assigned = oid

            # If no match, allocate a new ID
            if assigned is None:
                assigned = self.next_id
                self.next_id += 1

            prev_x  = self.tracked.get(assigned, {}).get('last_x', cx)
            counted = self.tracked.get(assigned, {}).get('counted', False)

            # If it crosses the center line, count and log
            if (not counted) and ((prev_x < self.center_x <= cx) or (prev_x > self.center_x >= cx)):
                self.total_cnt += 1
                counted = True
                self.class_cnts[label] = self.class_cnts.get(label, 0) + 1

                # Log timestamp and class label
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{timestamp}, {label}\n")
                log_file.flush()

            current[assigned] = {'last_x': cx, 'counted': counted}

            # Draw bounding box and label on frame
            color = self.class_colors[label]
            box_color = (int(color[2]), int(color[1]), int(color[0]))
            cv2.rectangle(
                frame,
                (x1, y1),
                (int(x1 + w0), int(y1 + h0)),
                box_color, 2
            )
            cv2.putText(
                frame, f"ID:{assigned} {label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                box_color, 2
            )

        # Update tracked dictionary for next frame
        self.tracked.clear()
        self.tracked.update(current)

        # Draw center line and count info on frame
        cv2.line(frame, (self.center_x, 0), (self.center_x, self.orig_h), (0, 0, 255), 2)
        cv2.putText(
            frame, f"Total: {self.total_cnt}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        yy = 60
        for lbl, cnt in self.class_cnts.items():
            cv2.putText(
                frame, f"{lbl}: {cnt}", (10, yy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2
            )
            yy += 30

        return frame

    def run(self):
        """
        Main loop: read frames from capture, run detection/tracking/counting,
        write annotated frames to output video, display in a window, and log counts.
        """
        # Open the count log file for this session
        with open(self.log_path, "w") as log_file:
            log_file.write("timestamp, class\n")  # CSV header

            try:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    annotated = self._detect_and_count(frame, log_file)
                    self.out.write(annotated)
                    cv2.imshow("JetCamCounter", annotated)

                    # Break on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            finally:
                # Release resources
                self.cap.release()
                self.out.release()
                cv2.destroyAllWindows()
                print(f"Done. Video saved to '{self.output_path}', log saved to '{self.log_path}'.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Object tracking and counting with YOLO (camera or video file input)"
    )
    parser.add_argument(
        'input', nargs='?', default=None,
        help='Input source. If omitted, camera (0) is used. Specify a file path to use a video file.'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    counter = JetCamCounter(input_source=args.input)
    counter.run()
