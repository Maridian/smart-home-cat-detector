from ultralytics import YOLO

class CatDetector:
    def __init__(self, model_name: str = "yolov8m.pt", conf_threshold: float = 0.35):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        
        # COCO Class IDs
        self.PERSON_CLASS_ID = 0
        self.CAT_CLASS_ID = 15

    def process_frame(self, frame):
        """
        Runs inference and returns:
        - should_save (bool): True if a cat is detected AND no human is in frame
        - best_conf (float): Highest confidence percentage for detected cat
        - annotated_frame: Frame with drawn bounding boxes for display
        """
        results = self.model(
            frame, 
            classes=[self.PERSON_CLASS_ID, self.CAT_CLASS_ID], 
            conf=self.conf_threshold, 
            verbose=False
        )
        boxes = results[0].boxes
        detected_classes = [int(box.cls[0]) for box in boxes] if len(boxes) > 0 else []

        has_cat = self.CAT_CLASS_ID in detected_classes
        has_human = self.PERSON_CLASS_ID in detected_classes

        best_conf = 0.0
        if has_cat:
            cat_confs = [float(box.conf[0]) for box in boxes if int(box.cls[0]) == self.CAT_CLASS_ID]
            best_conf = max(cat_confs) * 100

        annotated_frame = results[0].plot()
        should_save = has_cat and not has_human

        return should_save, best_conf, annotated_frame