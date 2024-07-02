
# Face Detection and YOLOv8 Training Script

This repository contains a Python script designed for capturing images of faces, creating a dataset, training a YOLOv8 model, and running real-time inference. The script is built using OpenCV, PyTorch, and the YOLO library from Ultralytics.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Creating the Dataset](#creating-the-dataset)
  - [Training the YOLOv8 Model](#training-the-yolov8-model)
  - [Running Inference](#running-inference)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/0xroyce/Yolo-v8-face-detection-and-training.git
   cd Yolo-v8-face-detection-and-training
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Creating the Dataset

1. Run the script to create a dataset:
   ```sh
   python script.py
   ```

2. Follow the prompts to enter the number of classes and the number of images for each class. Position your face in the frame and press 'c' to start capturing images. Press 'q' to quit capturing.

### Training the YOLOv8 Model

After creating the dataset, the script will automatically start training the YOLOv8 model.

### Running Inference

Once the model is trained, the script will begin running inference in real-time using your webcam. You can stop the inference by pressing 'q'.

## Dependencies

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO

You can install the dependencies by running:
```sh
pip install opencv-python torch ultralytics
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
