
# Human Detection in Drone Imagery for Search and Rescue Operations

## Introduction

In the wake of major disasters, timely and efficient search and rescue operations are crucial for saving lives. Traditional methods can be slow and may not cover all affected areas effectively. This project aims to enhance search and rescue efforts by detecting humans in drone imagery using advanced machine learning techniques. By leveraging drones equipped with real-time object detection capabilities, we can significantly improve the chances of locating survivors in disaster-stricken areas.

## Dataset

We utilize the [Lacmus Drone Dataset (LADD)](https://www.kaggle.com/datasets/mersico/lacmus-drone-dataset-ladd-v40) for training and testing our model. LADD provides a comprehensive collection of drone images specifically designed for human detection tasks, making it ideal for our project.

## Model

### YOLOv8m

For object detection and video processing, we have selected the [YOLOv8m model](https://docs.ultralytics.com/models/yolov8/). YOLOv8m offers a perfect balance between performance and computational efficiency, making it suitable for devices with limited processing power, such as mobile devices used in the field.

### Model Improvement

Our preliminary tests indicated that the pre-trained YOLOv8m model struggled to accurately detect humans who were far away in drone imagery. To address this limitation, we fine-tuned the model by augmenting it with additional data from LADD. This customization improved the model's accuracy in detecting distant individuals, which is crucial for effective search and rescue operations.

## Application Features

### Local Runtime

In disaster scenarios, network connectivity is often unreliable or nonexistent. Our application is designed to run locally on any iOS device, allowing drone operators to detect humans in real-time without the need for an internet connection. This local runtime capability ensures that search and rescue efforts are not hindered by connectivity issues.

### Video Upload and Storage

Users can upload and store videos within the application. This feature allows operators to:

- **Analyze Recorded Footage**: Review previously recorded videos for human detection, which is essential when covering large areas or when real-time analysis wasn't possible.
- **Data Archiving**: Keep a record of all flight footage for future reference or analysis.
- **Share Data**: Easily share important footage with other team members or authorities as needed.

## Installation

Please follow the steps below to install the application on your iOS device:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```
2. **Navigate to the Project Directory**
   ```bash
   cd your-repo-name
   ```
3. **Install Dependencies**
   - Ensure you have [Xcode](https://developer.apple.com/xcode/) installed.
   - Open the project in Xcode:
     ```bash
     open YourProject.xcodeproj
     ```


## Usage

### Real-Time Detection

1. **Launch the App** on your iOS device.
2. **Connect to the Drone**:
   - Ensure your drone is powered on and connected to your device.
3. **Start Live Feed**:
   - Navigate to the live feed section to view real-time video from the drone.
4. **Activate Detection**:
   - Tap on the **Detect** button to start human detection.
   - The app will highlight detected humans in the video feed.

### Uploading and Analyzing Videos

1. **Upload Video**:
   - Go to the **Videos** section in the app.
   - Tap on **Upload** to select a video from your device or drone storage.
2. **Analyze Footage**:
   - After uploading, select the video to analyze.
   - Tap on **Analyze** to run the human detection model on the selected footage.
3. **View Results**:
   - Detected humans will be highlighted in the video playback.
   - Use the timeline to navigate to specific timestamps where detections occurred.


## License

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE.

## Credits

This project uses code from [Ultralytics Yolo](https://github.com/ultralytics/yolo-ios-app) which is licensed under AGPL-3.0. Modifications were made to adapt it to our needs. The original work has been instrumental in helping us get started.


## Contact

For questions, suggestions, or collaboration inquiries, please contact us at:

- **Email**: [searchandrescue@example.com](mailto:searchandrescue@example.com)
- **GitHub Issues**: [GitHub Issues Page](https://github.com/yourusername/your-repo-name/issues)

---

*This README was generated to provide an overview of the Human Detection in Drone Imagery project aimed at improving search and rescue operations through advanced machine learning techniques.*
