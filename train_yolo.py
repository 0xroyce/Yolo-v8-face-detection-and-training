import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import time

current_dir = os.getcwd()


# Detect available devices
def get_available_devices():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA is available. Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return "cuda", num_gpus
    else:
        print("CUDA is not available. Using CPU.")
        return "cpu", 0


device, num_gpus = get_available_devices()
print(f"Using device: {device}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def capture_images(class_name, num_images):
    dataset_dir = os.path.join(current_dir, "dataset")
    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Capturing images for class '{class_name}'...")
    print("Position your face in the frame and press 'c' to start automatic capture, or 'q' to quit.")

    i = 0
    capture_started = False
    last_capture_time = 0
    capture_interval = 0.1
    output_size = 640

    augmentation = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.CLAHE(p=0.1),
        A.RandomShadow(p=0.2),
        A.RandomFog(p=0.1),
    ])

    while i < num_images:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if capture_started:
                cv2.putText(frame, f"Capturing: {i + 1}/{num_images}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Press 'c' to start capture", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                capture_started = True
                last_capture_time = time.time()
            elif key == ord('q'):
                print("\nCapture stopped.")
                break

            if capture_started and time.time() - last_capture_time >= capture_interval:
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face = frame[y:y + h, x:x + w]
                    resized = cv2.resize(face, (output_size, output_size))

                    augmented = augmentation(image=resized)
                    augmented_image = augmented["image"]

                    img_path = os.path.join(dataset_dir, "images", f"{class_name}_{i}.jpg")
                    cv2.imwrite(img_path, augmented_image)

                    label_path = os.path.join(dataset_dir, "labels", f"{class_name}_{i}.txt")
                    with open(label_path, "w") as f:
                        f.write(f"0 0.5 0.5 1.0 1.0")

                    print(f"\rProgress: {i + 1}/{num_images} images captured", end="", flush=True)
                    i += 1
                    last_capture_time = time.time()

    print("\nCapture completed.")
    cap.release()
    cv2.destroyAllWindows()


def create_dataset():
    num_classes = int(input("Enter the number of object classes: "))
    classes = []

    for i in range(num_classes):
        class_name = input(f"Enter name for class {i + 1}: ")
        classes.append(class_name)
        num_images = int(input(f"Enter number of images for {class_name}: "))
        capture_images(class_name, num_images)

    image_files = os.listdir(os.path.join(current_dir, "dataset", "images"))
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(current_dir, "dataset", "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "dataset", "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "dataset", "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "dataset", "val", "labels"), exist_ok=True)

    for file in train_files:
        os.rename(os.path.join(current_dir, "dataset", "images", file),
                  os.path.join(current_dir, "dataset", "train", "images", file))
        os.rename(os.path.join(current_dir, "dataset", "labels", file.replace(".jpg", ".txt")),
                  os.path.join(current_dir, "dataset", "train", "labels", file.replace(".jpg", ".txt")))

    for file in val_files:
        os.rename(os.path.join(current_dir, "dataset", "images", file),
                  os.path.join(current_dir, "dataset", "val", "images", file))
        os.rename(os.path.join(current_dir, "dataset", "labels", file.replace(".jpg", ".txt")),
                  os.path.join(current_dir, "dataset", "val", "labels", file.replace(".jpg", ".txt")))

    optimize_anchors(os.path.join(current_dir, "dataset", "train", "labels"))

    dataset_yaml_path = os.path.join(current_dir, "dataset.yaml")
    with open(dataset_yaml_path, "w") as f:
        f.write(f"train: {os.path.join(current_dir, 'dataset', 'train', 'images')}\n")
        f.write(f"val: {os.path.join(current_dir, 'dataset', 'val', 'images')}\n")
        f.write(f"nc: {num_classes}\n")
        f.write(f"names: {classes}")

    print(f"Dataset created and saved in {dataset_yaml_path}")


def optimize_anchors(labels_path):
    boxes = []
    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_path, label_file), 'r') as f:
                for line in f:
                    _, _, _, w, h = map(float, line.strip().split())
                    boxes.append([w, h])

    boxes = np.array(boxes)
    kmeans = KMeans(n_clusters=9, random_state=42)
    kmeans.fit(boxes)

    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors.prod(axis=1))]

    print("Optimized anchors:", anchors.tolist())
    return anchors


def train_yolo():
    print("Starting YOLOv8 training...")
    model = YOLO("yolov8x.pt")  # Use YOLOv8x for better performance
    dataset_yaml_path = os.path.join(current_dir, "dataset.yaml")

    # Determine batch size based on available resources
    if device == "cuda":
        batch_size = 16  # Reduced batch size
    else:
        batch_size = 8  # Reduced batch size for CPU training

    # Custom training settings
    try:
        results = model.train(
            data=dataset_yaml_path,
            epochs=1000,
            imgsz=640,
            patience=100,
            batch=batch_size,
            device=[0, 1],  # Use both GPUs
            workers=8,
            optimizer="AdamW",
            lr0=1e-3,
            lrf=1e-4,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            save=True,
            save_period=100,
            project=os.path.join(current_dir, "runs"),
            name="train",
            exist_ok=True,
            pretrained=True,
            amp=True,
            rect=True,
            multi_scale=True,
            close_mosaic=10
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        print("Attempting to train on CPU...")
        try:
            results = model.train(
                data=dataset_yaml_path,
                epochs=1000,
                imgsz=640,
                patience=100,
                batch=8,
                device="cpu",
                workers=8,
                optimizer="AdamW",
                lr0=1e-3,
                lrf=1e-4,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                pose=12.0,
                kobj=1.0,
                label_smoothing=0.0,
                nbs=64,
                overlap_mask=True,
                mask_ratio=4,
                dropout=0.0,
                val=True,
                save=True,
                save_period=100,
                project=os.path.join(current_dir, "runs"),
                name="train",
                exist_ok=True,
                pretrained=True,
                amp=False,  # Disable mixed precision for CPU
                rect=True,
                multi_scale=True,
                close_mosaic=10
            )
            print("Training completed successfully on CPU.")
        except Exception as e:
            print(f"Training failed on CPU as well. Error: {str(e)}")
            print("Please check your CUDA installation and GPU drivers.")


def run_inference():
    print("Starting inference...")
    weights_path = os.path.join(current_dir, "runs", "train", "weights", "best.pt")
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if ret:
            results = model(frame, conf=0.25, iou=0.45, max_det=10)  # Adjusted confidence and IoU thresholds
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")


if __name__ == "__main__":
    create_dataset()
    train_yolo()
    run_inference()