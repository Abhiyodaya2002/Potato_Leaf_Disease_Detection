# Potato Leaf Disease Detection
![Potato Leaf Disease](https://user-images.githubusercontent.com/67580321/171865245-b8f4a8c5-978b-4284-900b-3b7dd0b638a1.jpg)

## Problem Statement
Farmers cultivating potatoes often face severe financial losses due to diseases affecting their crops. Among the most common and devastating are **Early Blight** and **Late Blight**:
- **Early Blight**: Caused by the *Alternaria solani* fungus, this disease primarily affects older leaves, reducing yield significantly.
- **Late Blight**: Caused by the *Phytophthora infestans* microorganism, this disease spreads rapidly, especially under humid conditions, causing catastrophic damage.

Accurate and timely detection of these diseases can help farmers apply targeted treatments, preserve crop quality, and prevent significant economic losses. This project leverages **Deep Learning** and **Convolutional Neural Networks (CNNs)** to identify and classify potato leaf diseases with high accuracy.

---

## Project Description
This project introduces a deep learning-based solution for the agricultural domain, specifically focusing on potato leaf disease detection. Using a custom **CNN architecture**, the model can classify potato leaves into:
1. **Healthy**
2. **Early Blight**
3. **Late Blight**

### Key Features:
- **User-Friendly**: Easy-to-use interface for farmers or agricultural professionals.
- **High Accuracy**: Designed to provide reliable predictions for disease classification.
- **Remedy Suggestions**: Provides treatment options for identified diseases.

---

## Data Collection
The dataset for this project was sourced from Kaggle's **PlantVillage Dataset**. It includes labeled images of potato leaves across the three categories: healthy, early blight, and late blight. The dataset contains thousands of high-quality images suitable for training deep learning models.

**Dataset Link**: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/abdallahalidev/plantvillage-dataset)

---

## Architecture and Technologies Used

### Project Architecture:
1. **Data Preprocessing**:
   - Images resized and normalized for faster processing.
   - Data augmentation techniques (e.g., rotation, flipping) to enhance model generalization.

2. **Model Development**:
   - A custom **Convolutional Neural Network (CNN)** architecture designed for image classification.
   - Utilized **TensorFlow** and **Keras** libraries for model building and training.

3. **Model Training**:
   - Trained the model on GPU for faster computation.
   - Evaluated using metrics such as accuracy, precision, and recall.

4. **Deployment**:
   - Model integrated into a user-friendly web interface using **Flask**.
   - Option for users to upload images and receive predictions in real-time.

### Technologies Used:
- **Programming Languages**: Python
- **Deep Learning Libraries**: TensorFlow, Keras
- **Visualization Tools**: Matplotlib, Seaborn
- **Web Framework**: Flask
- **Dataset Source**: Kaggle
- **Development Environment**: Jupyter Notebook, VS Code

---

## How It Works
1. Upload an image of a potato leaf.
2. The trained CNN model processes the image.
3. Outputs the disease type (Healthy, Early Blight, Late Blight) and displays treatment recommendations.

---

## Live Demo
🎥 **[YouTube Link]**  
*Embed a video or link here showcasing the working of the project.*

---

## Installation and Usage
### Prerequisites:
- Python 3.7 or higher
- Libraries: TensorFlow, Keras, Flask, Matplotlib, NumPy, OpenCV

### Steps to Run the Project:
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Potato-Leaf-Disease-Detection.git
