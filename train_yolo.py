import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import albumentations as A
from sklearn.model_selection import train_test_split
import time

current_dir = os.getcwd()

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
    if class_name == "no_face":
        print("Capture images without faces. Press 'c' to start automatic capture, or 'q' to quit.")
    else:
        print("Position your face in the frame and press 'c' to start automatic capture, or 'q' to quit.")

    i = 0
    capture_started = False
    last_capture_time = time.time()
    capture_interval = 0.5  # Increase interval to 0.5 seconds for stability
    output_size = 640

    augmentation = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Blur(blur_limit=3, p=0.1),
        A.CLAHE(p=0.1),
        A.RandomShadow(p=0.2),
        A.RandomFog(p=0.1),
    ])

    while i < num_images:
        ret, frame = cap.read()
        if ret:
            current_time = time.time()
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            if class_name != "no_face":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if capture_started:
                cv2.putText(display_frame, f"Capturing: {i + 1}/{num_images}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Press 'c' to start capture", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Capture", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                capture_started = True
                last_capture_time = current_time
            elif key == ord('q'):
                print("\nCapture stopped.")
                break

            if capture_started and (current_time - last_capture_time) >= capture_interval:
                if class_name == "no_face":
                    resized = cv2.resize(frame, (output_size, output_size))
                    augmented = augmentation(image=resized)
                    augmented_image = augmented["image"]

                    img_path = os.path.join(dataset_dir, "images", f"{class_name}_{i}.jpg")
                    cv2.imwrite(img_path, augmented_image)

                    # Create an empty label file for negative samples
                    label_path = os.path.join(dataset_dir, "labels", f"{class_name}_{i}.txt")
                    open(label_path, 'w').close()  # This creates an empty file

                    i += 1
                elif len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y + h, x:x + w]

                    # Add padding to make the face image square
                    height, width = face_img.shape[:2]
                    max_dim = max(height, width)
                    top = (max_dim - height) // 2
                    bottom = max_dim - height - top
                    left = (max_dim - width) // 2
                    right = max_dim - width - left
                    face_img_padded = cv2.copyMakeBorder(face_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                         value=[0, 0, 0])

                    resized = cv2.resize(face_img_padded, (output_size, output_size))
                    augmented = augmentation(image=resized)
                    augmented_image = augmented["image"]

                    img_path = os.path.join(dataset_dir, "images", f"{class_name}_{i}.jpg")
                    cv2.imwrite(img_path, augmented_image)

                    label_path = os.path.join(dataset_dir, "labels", f"{class_name}_{i}.txt")
                    with open(label_path, "w") as f:
                        f.write(f"0 0.5 0.5 1.0 1.0")  # Full image as we're saving only the face

                    i += 1

                last_capture_time = current_time
                print(f"\rProgress: {i}/{num_images} images captured", end="", flush=True)

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

    # Add negative samples
    num_negative = int(input("Enter number of negative samples (images without faces): "))
    capture_images("no_face", num_negative)

    image_files = [f for f in os.listdir(os.path.join(current_dir, "dataset", "images")) if f.endswith('.jpg')]
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(current_dir, "dataset", "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "dataset", "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "dataset", "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "dataset", "val", "labels"), exist_ok=True)

    def move_files(file_list, destination):
        for file in file_list:
            image_src = os.path.join(current_dir, "dataset", "images", file)
            label_src = os.path.join(current_dir, "dataset", "labels", file.replace(".jpg", ".txt"))

            image_dst = os.path.join(destination, "images", file)
            label_dst = os.path.join(destination, "labels", file.replace(".jpg", ".txt"))

            os.rename(image_src, image_dst)
            if os.path.exists(label_src):
                os.rename(label_src, label_dst)

    move_files(train_files, os.path.join(current_dir, "dataset", "train"))
    move_files(val_files, os.path.join(current_dir, "dataset", "val"))

    dataset_yaml_path = os.path.join(current_dir, "dataset.yaml")
    with open(dataset_yaml_path, "w") as f:
        f.write(f"train: {os.path.join(current_dir, 'dataset', 'train', 'images')}\n")
        f.write(f"val: {os.path.join(current_dir, 'dataset', 'val', 'images')}\n")
        f.write(f"nc: {num_classes}\n")
        f.write(f"names: {classes}")

    print(f"Dataset created and saved in {dataset_yaml_path}")

def train_yolo():
    print("Starting YOLOv8 training for face detection...")
    model = YOLO("yolov8m.pt")  # Using medium model
    dataset_yaml_path = os.path.join(current_dir, "dataset.yaml")

    try:
        results = model.train(
            data=dataset_yaml_path,
            epochs=300,
            imgsz=640,
            patience=50,
            batch=16,
            device=0 if num_gpus > 0 else "cpu",
            workers=8,
            verbose=True,
            seed=0,
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.1,
            nbs=64,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.8,
            mixup=0.1,
            copy_paste=0.1,
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        results = None

    return results

def run_inference():
    print("Starting inference...")
    weights_path = os.path.join(current_dir, "runs", "detect", "train", "weights", "best.pt")
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if ret:
            results = model(frame, conf=0.5, iou=0.45)  # Increased confidence threshold
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Face Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")

if __name__ == "__main__":
    create_dataset()
    #train_yolo()
    #run_inference()