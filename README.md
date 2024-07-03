
# Face Detection and YOLOv8 Training Script

This repository contains a Python script designed for capturing images of faces, creating a dataset, training a YOLOv8 model, and running real-time inference. The script is built using OpenCV, PyTorch, and the YOLO library from Ultralytics.

The script works on CPU or GPU(s) but I recommend at least 1 GPU.

I recommend to set 250 images to capture as a minimum and 80 negative images as a minimum. In ideal scenario, you capture +600 images and +180 negative images. But bear in mind that you will need to sit in front of the camera for quite a long time. You can, however, shorten the capture images time and test it.

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
   python train_yolo.py
   ```

2. Follow the prompts to enter the number of classes and the number of images for each class. Position your face in the frame and press 'c' to start capturing images. Press 'q' to quit capturing.

### Training the YOLOv8 Model

After creating the dataset, the script will automatically start training the YOLOv8 model.

### Running Inference

Once the model is trained, the script will begin running inference in real-time using your webcam. You can stop the inference by pressing 'q'.

## Dependencies

You can install the dependencies by running:
```sh
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
