import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import time


class BacteriaDataset(Dataset):
    """Dataset for bacteria images"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # For training with labels, you would return (image, label)
        # For inference, just return the image
        return image


class BacteriaDetectionModel(nn.Module):
    """CNN model for bacteria detection"""
    def __init__(self, num_classes=2):  # 2 classes: bacteria present or not
        super(BacteriaDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Adjust according to your input size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BacteriaSegmentationModel(nn.Module):
    """U-Net style model for bacteria segmentation"""
    def __init__(self):
        super(BacteriaSegmentationModel, self).__init__()
        # Encoder
        self.enc1 = self._make_layer(3, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        # Decoder
        self.dec1 = self._make_layer(512, 256)
        self.dec2 = self._make_layer(256, 128)
        self.dec3 = self._make_layer(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoding
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoding with skip connections
        dec1 = self.dec1(self.upsample(enc4))
        dec2 = self.dec2(self.upsample(dec1 + enc3))
        dec3 = self.dec3(self.upsample(dec2 + enc2))
        
        output = torch.sigmoid(self.final(dec3 + enc1))
        return output


class ImagePreprocessor:
    """Class for preprocessing microscope images"""
    def __init__(self):
        self.mp_hands = mp.solutions.hands  # Will be used for scale reference if needed
        
    def preprocess_image(self, image):
        """
        Preprocess image for better bacteria detection
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to highlight bacteria
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Use morphological operations to enhance bacteria shapes
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Convert back to RGB for compatibility with the model
        processed = cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB)
        
        return processed
    
    def enhance_contrast(self, image):
        """Enhance contrast using CLAHE"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        return enhanced


class BacteriaAnalyzer:
    """Main class for bacteria detection and analysis"""
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()
        
        # Initialize model
        self.model = BacteriaSegmentationModel().to(self.device)
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_path}")
        else:
            print("No model loaded. Using untrained model.")
            
        # Image transformation for the model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def detect_bacteria(self, image):
        """
        Detect bacteria in the given image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            detected_image: Image with bacteria highlighted
            count: Estimated bacteria count
            coverage: Percentage of area covered by bacteria
        """
        # Preprocess the image
        processed = self.preprocessor.preprocess_image(image)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        
        # Apply transformations
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Convert output to binary mask
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Resize mask to original image size
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Find contours (bacteria)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on original image
        detected_image = image.copy()
        cv2.drawContours(detected_image, contours, -1, (0, 255, 0), 2)
        
        # Count bacteria and calculate coverage
        count = len(contours)
        total_area = image.shape[0] * image.shape[1]
        bacteria_area = sum(cv2.contourArea(c) for c in contours)
        coverage = (bacteria_area / total_area) * 100
        
        return detected_image, count, coverage
    
    def train_model(self, train_dir, val_dir, epochs=10, batch_size=8, learning_rate=0.001):
        """
        Train the bacteria detection model
        
        Args:
            train_dir: Directory containing training images
            val_dir: Directory containing validation images
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
        Returns:
            history: Training history
        """
        # Set model to training mode
        self.model.train()
        
        # Create datasets and dataloaders
        train_dataset = BacteriaDataset(train_dir, transform=self.transform)
        val_dataset = BacteriaDataset(val_dir, transform=self.transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            self.model.train()
            for inputs in train_loader:
                inputs = inputs.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)  # Using input as target for demonstration
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            # Validation
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for inputs in val_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item() * inputs.size(0)
            
            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}')
        
        return history
    
    def save_model(self, path):
        """Save model to the specified path"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


class BacteriaApp:
    """Main application class for bacteria detection"""
    def __init__(self, model_path=None):
        self.analyzer = BacteriaAnalyzer(model_path)
        self.cap = None
        
    def start_camera(self, camera_id=0):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
        return True
    
    def process_video_feed(self):
        """Process video feed from camera"""
        if not self.cap:
            print("Error: Camera not initialized.")
            return
            
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break
                
            # Process frame for bacteria detection
            start_time = time.time()
            detected_image, count, coverage = self.analyzer.detect_bacteria(frame)
            fps = 1.0 / (time.time() - start_time)
            
            # Display results on frame
            cv2.putText(detected_image, f"Bacteria Count: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(detected_image, f"Coverage: {coverage:.2f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(detected_image, f"FPS: {fps:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show result
            cv2.imshow("Bacteria Detection", detected_image)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path):
        """Process a single image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return
            
        # Process image
        detected_image, count, coverage = self.analyzer.detect_bacteria(image)
        
        # Display results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Bacteria (Count: {count}, Coverage: {coverage:.2f}%)")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    def batch_process(self, input_dir, output_dir):
        """Process all images in a directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        
        results = []
        
        for img_file in image_files:
            print(f"Processing {img_file}...")
            img_path = os.path.join(input_dir, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Error: Could not read image from {img_path}")
                continue
                
            # Process image
            detected_image, count, coverage = self.analyzer.detect_bacteria(image)
            
            # Save result
            output_path = os.path.join(output_dir, f"detected_{img_file}")
            cv2.imwrite(output_path, detected_image)
            
            # Store results
            results.append({
                'filename': img_file,
                'bacteria_count': count,
                'coverage_percent': coverage
            })
            
        # Generate report
        report_path = os.path.join(output_dir, "detection_report.txt")
        with open(report_path, 'w') as f:
            f.write("Bacteria Detection Report\n")
            f.write("=======================\n\n")
            
            for result in results:
                f.write(f"File: {result['filename']}\n")
                f.write(f"Bacteria Count: {result['bacteria_count']}\n")
                f.write(f"Coverage: {result['coverage_percent']:.2f}%\n")
                f.write("-----------------------\n")
                
        print(f"Batch processing complete. Results saved to {output_dir}")
        print(f"Report generated at {report_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bacteria Detection Application")
    parser.add_argument("--mode", choices=["camera", "image", "batch", "train"],
                        default="camera", help="Application mode")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the trained model")
    parser.add_argument("--input", type=str, default=None,
                        help="Input image path or directory")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory for batch processing")
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Training data directory")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Validation data directory")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if not args.train_dir or not args.val_dir:
            print("Error: Training and validation directories must be specified.")
            return
            
        analyzer = BacteriaAnalyzer()
        history = analyzer.train_model(args.train_dir, args.val_dir, epochs=args.epochs)
        
        # Save the model
        if not os.path.exists("models"):
            os.makedirs("models")
        analyzer.save_model("models/bacteria_model.pth")
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.savefig("models/training_history.png")
        plt.show()
        
    else:
        app = BacteriaApp(args.model)
        
        if args.mode == "camera":
            if app.start_camera():
                app.process_video_feed()
                
        elif args.mode == "image":
            if not args.input:
                print("Error: Input image path must be specified.")
                return
            app.process_image(args.input)
            
        elif args.mode == "batch":
            if not args.input:
                print("Error: Input directory must be specified.")
                return
            app.batch_process(args.input, args.output)


if __name__ == "__main__":
    main()
