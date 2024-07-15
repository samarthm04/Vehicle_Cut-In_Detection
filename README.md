
# Vehicle Cut-in and Collision Detection for Indian Roads

## Overview
![Untitled](https://github.com/user-attachments/assets/9559915f-26e9-4970-bf1f-d4bbcb7a2633)

This project implements a robust system for vehicle cut-in and collision detection specifically tailored for the challenging and unstructured road environments typical of countries like India. Traditional lane tracking systems often fail in such conditions due to the lack of clear lane markings and the chaotic nature of the traffic. Our system addresses these issues by leveraging a fine-tuned YOLOv8 model for object detection, custom methodologies for dynamic lane area creation, and a sophisticated approach to Time to Collision (TTC) calculation. Despite the complexities of disorganized roads and limited computational resources, the project demonstrates significant potential for improving road safety.

While precision scores and other typical performance metrics provide some insight into the model's effectiveness, they do not fully capture the robustness and versatility of our approach. Our focus is on accurately detecting objects that enter the region of interest, irrespective of their tags, ensuring timely and reliable warnings. This makes the system highly adaptable to a variety of unstructured environments, providing critical safety features where they are most needed.


## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Methodology](#methodology)
  - [Object Detection](#object-detection)
  - [Time to Collision (TTC) Calculation](#time-to-collision-ttc-calculation)
  - [Lane Creation](#lane-creation)

  - [Cut-in Detection](#cut-in-detection)
- [Challenges Faced](#challenges-faced)
- [Technologies Used](#technologies-used)
- [Datasets](#datasets)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Introduction
The goal of this project is to develop a reliable vehicle cut-in and collision detection system that functions effectively even on roads without clear lane markings. Traditional lane tracking systems often fail in such conditions, making this approach particularly suitable for Indian roads, where lane markings are sparse or nonexistent.

## Features
- **Dynamic Lane Area Creation:** Uses color segmentation to define road and sky regions, dynamically creating a lane area that adapts to different viewpoints.
- **Object Detection:** Utilizes a fine-tuned YOLOv8 (You Only Look Once) model for detecting vehicles and other objects on the road. YOLO is a state-of-the-art, real-time object detection system known for its speed and accuracy.
- **Time to Collision (TTC) Calculation:** Implements a robust methodology to calculate TTC, considering relative velocity and acceleration.
- **Cut-in Detection:** Identifies vehicles that cut into the dynamically defined lane area, providing timely warnings.
- **Adaptability:** Designed to work in unstructured road environments with minimal reliance on lane markings.

## Methodology

### Object Detection
The system employs a YOLOv8 model fine-tuned on a custom dataset to detect vehicles and other relevant objects. YOLOv8 is an advanced version of the YOLO(You Only Look Once) series, which is known for its efficiency and high detection accuracy, making it suitable for real-time applications on Indian roads.


### Time to Collision (TTC) Calculation
1. **Object Detection:** YOLOv8 model identifies objects and their bounding boxes.
2. **Distance Estimation:** Object width is used to estimate its distance from the camera.
3. **Velocity Calculation:** Object velocity is determined based on its movement across frames.
4. **Acceleration Consideration:** The system factors in acceleration for more precise TTC estimation.
5. **Warning Generation:** If the calculated TTC drops below 0.6 seconds and the object is within the dynamically defined lane area, the system generates a warning.

### Rationale for Using 0.6 Seconds:
The choice of 0.6 seconds as the TTC threshold is based on human reaction time considerations. Research indicates that 0.6 seconds is an optimal threshold because it aligns closely with the average human reaction time to unexpected events on the road. This threshold strikes a balance: it provides enough time for drivers to react and take evasive action without triggering unnecessary false alarms. Therefore, it ensures that warnings are issued precisely when needed, enhancing the system's reliability in real-world scenarios.



### Lane Creation
- **Road and Sky Segmentation:** HSV (Hue, Saturation, Value) color space is used to segment road and sky regions. HSV is particularly effective in varying lighting conditions, which is common on unstructured roads.

- **Stable Line Detection:** Stable lines are identified within the road and sky regions.
- **Trapezium (Region of Interest) Creation:** A dynamic lane area is formed as a trapezium using the detected lines.

### Cut-in Detection
Cut-in detection is achieved by monitoring objects that enter the dynamically defined lane area. The system calculates the intersection area between detected objects and the lane area. If the intersection area exceeds a threshold, a cut-in is detected, and appropriate warnings are generated.


## Challenges Faced
- **Disorganized Roads:** Overcoming the difficulties of lane tracking without clear lane markings.
- **GPU Limitations:** Addressing constraints in training large datasets due to limited GPU resources.
- **Data Storage:** Managing large datasets within limited storage capacity.
- **Parameter Tuning:** Identifying optimal parameters for object detection, segmentation, and TTC calculation.
- **Colab Limitations:** Mitigating session timeouts and GPU availability issues on Google Colab.

## Technologies Used
- **Python:** Core programming language for implementation.
- **OpenCV:** Library for image processing and computer vision tasks.
- **YOLOv8:** Deep learning model for object detection.
- **Shapely:** Library for geometric operations, used for intersection calculations.
- **Numpy:** Library for numerical computations.
- **Base64 and HTML:** Used for creating downloadable video links.


## Datasets
The project utilized a custom dataset named **Dataset-IDD Detection**, consisting of images and videos from Indian roads. Data augmentation techniques were employed to enhance the dataset size and diversity.

### Dataset Details
- **Train Images:** 4955
- **Train Annotations:** 11577
- **Validation Images:** 1014
- **Validation Annotations:** 2028

**Dataset Link:** [Download Dataset](https://idd.insaan.iiit.ac.in/dataset/download/)

This dataset provides a comprehensive set of images and annotations that capture the complexity and variety of Indian road conditions. It includes diverse scenarios with varying lighting, weather conditions, and road structures, making it ideal for training and validating the vehicle cut-in and collision detection model.


## Setup
1. **Prerequisites:**
   - Python 3.6 or later
   - OpenCV
   - YOLOv8 
   - Shapely
   - NumPy
2. **Installation:**
   ```bash
   pip install ultralytics
   pip install shapely
   pip install numpy
   pip install opencv-python

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
-Precision: 0.85
-Recall: 0.8
-F1-Score: 0.82

### Reflections on Performance Metrics
While traditional performance metrics like precision and recall are important, they do not fully capture the robustness and versatility of our system. Our primary focus is on detecting any object that enters the region of interest, regardless of its tag. This means that even if an object is tagged incorrectly, as long as it is detected and the TTC and cut-in criteria are met, the system will provide the necessary warnings. This makes our model extremely robust and effective in real-world scenarios where perfect tagging is not always possible.

### Comparison with Pre-trained Model
The performance of the custom fine-tuned model demonstrates significant improvements over the pre-trained model in terms of detection accuracy and reliability in unstructured road conditions. For detailed performance analysis, refer to pre-trained_model_performance.ipynb.

![41045d8b-cb50-4869-a151-f1191e18d58a](https://github.com/user-attachments/assets/6b7cb31d-e8fa-4251-b3a2-70e28fb3d4bd)


## Model Speciality

### Robustness and Versatility
- **Region of Interest (ROI) Focused:** The model is designed to focus on the region of interest, which is dynamically defined by a trapezium. This ensures that any object entering this area is detected, regardless of its specific tag.
- **Dynamic Lane Area Creation:** The use of color segmentation to create a dynamic lane area allows the model to adapt to varying road conditions, ensuring reliable performance even on roads without clear lane markings.
- **TTC and Cut-in Detection:** The primary goal is to provide timely warnings for potential collisions. As long as the object is detected in the ROI and the TTC criteria are met, the system will issue a warning. This makes the model effective in preventing collisions, even if object classification is not perfect.

### Metrics vs. Real-World Performance
- **Precision Scores:** Traditional precision metrics may not fully reflect the model's effectiveness in real-world scenarios. Misclassifications may occur, but what truly matters is the detection of objects entering the ROI and the accurate calculation of TTC.
- **Adaptability:** The model's ability to work in various lighting and weather conditions, and its focus on the ROI, ensures that it provides reliable warnings in diverse environments. This adaptability is crucial for deployment in unstructured road environments typical of countries like India.

### Advantages Over Traditional Systems


- **Independence from Lane Markings:** Unlike traditional systems that rely heavily on lane markings, our model dynamically defines lane areas, making it suitable for disorganized roads.
- **Real-Time Warnings:** The model provides real-time warnings for potential collisions, enhancing road safety by allowing timely interventions.

## Conclusion
This project demonstrates a robust approach to vehicle cut-in and collision detection in challenging road environments. By leveraging pre-trained models and innovative methodologies, the system overcomes the limitations of traditional lane tracking and provides reliable performance even in the absence of lane markings.

## Future Work
- **Model Improvement:** Further fine-tuning and training with larger datasets to enhance detection accuracy.
- **Real-time Implementation:** Optimization for real-time performance on embedded systems.
- **Enhanced Parameter Tuning:** Automated parameter tuning using machine learning techniques.
- **Additional Features:** Integration of other safety features like pedestrian detection and traffic sign recognition.
- **Leveraging Sensors:** Incorporating additional sensors like LIDAR to improve accuracy and robustness, complementing the current 2D camera data. This can help in better distance estimation and object detection under various conditions.

## Specific Failure Cases
Occlusion: The model might fail to detect vehicles that are partially occluded by other objects.
Adverse Weather Conditions: Heavy rain, fog, or snow can affect the model's detection capabilities.
Nighttime Driving: Reduced visibility at night might lead to decreased detection accuracy.
Complex Road Scenarios: Highly congested or complex traffic scenarios may challenge the model's ability to accurately track multiple vehicles.
Repository Information
The main code for the project is in the GitHub repository under the file name collision_detection+cut_in.ipynb. The data counting script can be found in data_counter.ipynb, and the performance evaluation of the pre-trained model is in pre-trained_model_performance.ipynb.
