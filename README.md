
# Vehicle Cut-in and Collision Detection

## Overview
This project implements a robust system for vehicle cut-in and collision detection specifically tailored for the challenging and unstructured road environments typical of countries like India. The system leverages a fine-tuned YOLOv8 model for object detection, custom methodologies for dynamic lane area creation, and a sophisticated approach to Time to Collision (TTC) calculation. Despite the complexities of disorganized roads and limited computational resources, the project demonstrates significant potential for improving road safety.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Methodology](#methodology)
    - [Object Detection](#object-detection)
    - [Time to Collision (TTC) Calculation](#time-to-collision-ttc-calculation)
    - [Lane Creation](#lane-creation)
    - [Cut-in Detection](#cut-in-detection)
4. [Challenges Faced](#challenges-faced)
5. [Technologies Used](#technologies-used)
6. [Dataset-IDD Detection](#dataset-idd-detection)
7. [Setup](#setup)
8. [Usage](#usage)
9. [Results](#results)
10. [Conclusion](#conclusion)
11. [Future Work](#future-work)

## Introduction
The goal of this project is to develop a reliable vehicle cut-in and collision detection system that functions effectively even on roads without clear lane markings. Traditional lane tracking systems often fail in such conditions, making this approach particularly suitable for Indian roads, where lane markings are sparse or nonexistent.

## Features
- **Dynamic Lane Area Creation:** Uses color segmentation to define road and sky regions, dynamically creating a lane area that adapts to different viewpoints.
- **Object Detection:** Utilizes a fine-tuned YOLOv8 model for detecting vehicles and other objects on the road.
- **Time to Collision (TTC) Calculation:** Implements a robust methodology to calculate TTC, considering relative velocity and acceleration.
- **Cut-in Detection:** Identifies vehicles that cut into the dynamically defined lane area, providing timely warnings.
- **Adaptability:** Designed to work in unstructured road environments with minimal reliance on lane markings.

## Methodology

### Object Detection
The system employs a YOLOv8 model fine-tuned on a custom dataset to detect vehicles and other relevant objects. This approach ensures high detection accuracy tailored to the specific characteristics of Indian roads.

### Time to Collision (TTC) Calculation
The TTC is calculated using the detected objects' positions and velocities. Here's a breakdown of the process:
1. **Object Detection:** The YOLOv8 model provides bounding boxes for detected objects.
2. **Distance Estimation:** The width of the detected object is used to estimate its distance from the camera.
3. **Velocity Calculation:** The velocity of the object is calculated based on its movement between frames.
4. **Acceleration Consideration:** The system accounts for acceleration, providing a more accurate TTC.
5. **Warning Generation:** If the TTC falls below a predefined threshold and the object is within the defined lane area, a warning is generated.

### Lane Creation
To address the lack of lane markings, the system dynamically defines a lane area using color segmentation:
1. **Road and Sky Segmentation:** HSV color space is used to segment road and sky regions.
2. **Stable Line Detection:** The system identifies stable road and sky lines to define the lane area.
3. **Trapezium Creation:** A trapezium is drawn based on these lines, creating a dynamic lane area that adapts to different viewpoints.

### Cut-in Detection
Cut-in detection is achieved by monitoring objects that enter the dynamically defined lane area. The system calculates the intersection area between detected objects and the lane area. If the intersection area exceeds a threshold, a cut-in is detected, and appropriate warnings are generated.

## Challenges Faced
1. Disorganized Roads and Lack of Lane Markings: Implementing effective lane tracking in the absence of lane markings.
2. GPU Limitations: Difficulty in training large datasets due to limited access to powerful GPUs.
3. Data Storage: Managing large datasets with limited storage capacity.
4. Parameter Tuning: Finding optimal parameters for object detection, road segmentation, and TTC calculation.
5. Colab Limitations: Struggles with session timeouts and GPU availability on Google Colab, leading to the decision to run Jupyter notebooks natively.

## Technologies Used
- **Python:** Core programming language for implementation.
- **OpenCV:** Library for image processing and computer vision tasks.
- **YOLOv8:** Deep learning model for object detection.
- **Shapely:** Library for geometric operations, used for intersection calculations.
- **Numpy:** Library for numerical computations.
- **Base64 and HTML:** Used for creating downloadable video links.

## Dataset-*IDD Detection*
The project utilized a custom dataset consisting of images and videos from Indian roads. Data augmentation techniques were employed to enhance the dataset size and diversity.

### Dataset Details
- **Train Images:** 4955
- **Train Annotations:** 11577
- **Validation Images:** 1014
- **Validation Annotations:** 2028

**Dataset Link:** [Download Dataset](https://idd.insaan.iiit.ac.in/dataset/download/)

This dataset provides a comprehensive set of images and annotations that capture the complexity and variety of Indian road conditions. It includes diverse scenarios with varying lighting, weather conditions, and road structures, making it ideal for training and validating the vehicle cut-in and collision detection model.

## Setup

### Prerequisites
- Python 3.6 or later
- OpenCV
- YOLOv8
- Shapely
- Numpy

### Installation
```bash
pip install ultralytics
pip install shapely
pip install numpy
pip install opencv-python
```

## Usage

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/samarthm04/Vehicle_Cut-In_Detection.git
    cd vehicle-cut-in-and-collision-detection
    ```

2. **Run the Script:**
    ```bash
    python main.py
    ```

3. **Output Video:**
   After processing, the output video will be saved as `output_video.mp4`.

## Results
The system effectively detects vehicle cut-ins and calculates TTC, providing timely warnings in unstructured road environments. The dynamic lane area creation method allows the system to function without reliance on lane markings, making it highly adaptable.

### Model Performance Metrics
- **Precision:** 0.85
- **Recall:** 0.8
- **F1-Score:** 0.82

### Comparison with Pre-trained Model
The performance of the custom fine-tuned model demonstrates significant improvements over the pre-trained model in terms of detection accuracy and reliability in unstructured road conditions. For detailed performance analysis, refer to `pre-trained_model_performance.ipynb`.

![41045d8b-cb50-4869-a151-f1191e18d58a](https://github.com/user-attachments/assets/7e036915-c274-4ac5-8297-483b55c2c464)

## Conclusion
This project demonstrates a robust approach to vehicle cut-in and collision detection in challenging road environments. By leveraging pre-trained models and innovative methodologies, the system overcomes the limitations of traditional lane tracking and provides reliable performance even in the absence of lane markings.

## Future Work
- **Model Improvement:** Further fine-tuning and training with larger datasets to enhance detection accuracy.
- **Real-time Implementation:** Optimization for real-time performance on embedded systems.
- **Enhanced Parameter Tuning:** Automated parameter tuning using machine learning techniques.
- **Additional Features:** Integration of other safety features like pedestrian detection and traffic sign recognition.
- **Leveraging Sensors:** Incorporating additional sensors like LIDAR to improve accuracy and robustness, complementing the current 2D camera data. This can help in better distance estimation and object detection under various conditions.

## Specific Failure Cases
- **Occlusion:** The model might fail to detect vehicles that are partially occluded by other objects.
- **Adverse Weather Conditions:** Heavy rain, fog, or snow can affect the model's detection capabilities.
- **Nighttime Driving:** Reduced visibility at night might lead to decreased detection accuracy.
- **Complex Road Scenarios:** Highly congested or complex traffic scenarios may challenge the model's ability to accurately track multiple vehicles.

## Repository Information
The main code for the project is in the GitHub repository under the file name `collision_detection+cut_in.ipynb`. The data counting script can be found in `data_counter.ipynb`, and the performance evaluation of the pre-trained model is in `pre-trained_model_performance.ipynb`.

