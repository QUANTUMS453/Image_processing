import cv2
from ultralytics import YOLO


class Detection:
    def __init__(self, vid_path, model_name):
        self.model = YOLO(model_name)  # Load model
        self.model.to("cuda")
        self.vid = cv2.VideoCapture(vid_path)  # Open video
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter("output.mp4", self.fourcc, 30.0, (1920, 1080))  # Save output

    def annotate_frame(self, frame, show=False):
        results = self.model(frame)  # Perform YOLO detection
        annotated_frame = results[0].plot()  # Draw bounding boxes
        self.show(annotated_frame)
        return annotated_frame

    def show(self, frame):
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Detection", 1280, 720)
        cv2.imshow("YOLO Detection", frame)

    def run(self, show=False):
        width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.vid.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter("output.mp4", self.fourcc, fps, (width, height))

        while self.vid.isOpened():
            ret, frame = self.vid.read()
            if not ret:
                break
            annotated_frame = self.annotate_frame(frame)

            if show:
                self.show(annotated_frame)
                

            self.out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.vid.release()
        self.out.release()
        cv2.destroyAllWindows()
