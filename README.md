Bacteria Detection 
---
![Pipeline](assets/demo.gif)

Overview
---
This application is designed for detecting and analyzing bacteria in microscope images or real-time video feeds. It uses a U-Net style convolutional neural network (CNN) for bacteria segmentation, combined with image preprocessing techniques to enhance detection accuracy. The application supports multiple modes: real-time camera feed processing, single image analysis, batch image processing, and model training.

Features
---
* Real-time Bacteria Detection: Process video feeds from a camera to detect and count bacteria with real-time visualization.
* Single Image Processing: Analyze individual microscope images to identify bacteria and calculate coverage.
* Batch Processing: Process multiple images in a directory and generate a detailed detection report.
* Model Training: Train the bacteria segmentation model using labeled datasets.
* Preprocessing: Enhance images using techniques like Gaussian blur, adaptive thresholding, and CLAHE for better detection.

Requirements
---
```
pip install torch torchvision opencv-python numpy matplotlib mediapipe pillow
```

Installation
---
1. Clone the repository:
```
git clone https://github.com/your-repo/bacteria-detection.git
cd bacteria-detection
```
2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the required dependencies:
```
pip install torch torchvision opencv-python numpy matplotlib mediapipe pillow
```

Usage
---
The application supports four modes: camera, image, batch, and train. Run the script using the command-line interface.

Command-Line Arguments
---
* --mode: Operation mode (camera, image, batch, or train). Default: camera.
* --model: Path to a pre-trained model file (optional).
* --input: Path to input image or directory (required for image and batch modes).
* --output: Output directory for batch processing results. Default: output.
* --train_dir: Directory containing training images (required for train mode).
* --val_dir: Directory containing validation images (required for train mode).
* --epochs: Number of training epochs (default: 10).

Examples
---
1. Real-time Camera Processing:
```
python detection.py --mode camera --model models/bacteria_model.pth
```
This starts the camera feed, detects bacteria, and displays the count, coverage, and FPS in real-time. Press q to exit.

2. Single Image Processing:
```
python detection.py --mode image --input path/to/image.jpg --model models/bacteria_model.pth
```
This processes a single image and displays the original and detected images side-by-side with bacteria count and coverage.

3. Batch Processing:
```
python detection.py --mode batch --input path/to/input_dir --output path/to/output_dir --model models/bacteria_model.pth
```
This processes all images in the input directory, saves detected images to the output directory, and generates a detection_report.txt with results.

4. Training the Model:
```
python detection.py --mode train --train_dir path/to/train_dir --val_dir path/to/val_dir --epochs 10
```
This trains the model, saves it to models/bacteria_model.pth, and plots the training history.

Notes
---
* The application assumes input images are in .png, .jpg, .jpeg, or .tif format.
* For training, ensure the training and validation directories contain properly labeled data (currently uses input images as targets for demonstration).
* The model expects input images to be resized to 224x224 pixels and normalized using ImageNet statistics.
* Ensure a CUDA-compatible GPU is available for faster processing; otherwise, it defaults to CPU.
