
# COVID-19 Image Classification using Custom VGG16 Model

This project demonstrates how to classify CT scan images of COVID-19 and non-COVID (normal) patients using a custom Convolutional Neural Network (CNN) based on the VGG16 architecture.

## Key Steps

### 1. **Data Preprocessing**
   - The dataset contains 7,593 COVID-19 images and 6,893 normal images.
   - 5,000 images from each class (COVID and normal) are randomly sampled for training.
   - Images are resized to 256x256 pixels to reduce memory usage while retaining relevant features.
   - Labels are created for the images: `0` for normal images and `1` for COVID-19 images.

### 2. **Image Visualization**
   - Random samples from the COVID-19 and normal CT scans are displayed to provide a quick look at the dataset.

### 3. **Model Architecture**
   - A custom VGG16 model is used for classification:
     - The model includes layers like Conv2D, MaxPooling2D, Dropout, and Dense layers.
     - The model architecture is designed to classify between two classes: `normal` and `covid`.

### 4. **Model Training**
   - The training set is split into training and validation sets using `train_test_split`.
   - The model is trained using the Adam optimizer and binary cross-entropy loss function.
   - Two callbacks are used during training:
     1. **Early Stopping**: Prevents overfitting by stopping the training if validation loss does not improve.
     2. **Model Checkpoint**: Saves the model whenever the validation loss is minimized.

### 5. **Model Evaluation**
   - After training, the model is evaluated on a separate test set (1,500 COVID and 1,500 normal images).
   - **Classification Report**: Precision, Recall, and F1-Score are used to evaluate the model's performance, in addition to accuracy and loss metrics.

### 6. **Metrics and Results**
   - The model's classification performance is assessed using:
     - **Precision**: Percentage of correct positive predictions out of all positive predictions.
     - **Recall**: Percentage of correct positive predictions out of all actual positives.
     - **F1-Score**: Harmonic mean of precision and recall, providing a balanced performance metric.
   - Results are plotted showing training and validation accuracy and loss.

## Libraries Used
- `numpy`: For numerical operations.
- `pandas`: For data processing.
- `cv2`: For reading and resizing images.
- `tensorflow`: For building and training the neural network model.
- `sklearn`: For model evaluation and splitting the data.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- scikit-learn
- Matplotlib

## Usage
1. Clone the repository and install the required dependencies.
2. Place the COVID-19 CT scan images and normal CT scan images in the appropriate folders.
3. Run the code to start training the model.
4. Once the model is trained, you can evaluate it on a test set and visualize the results.

## Conclusion
The custom VGG16 model performs reasonably well in classifying COVID-19 and non-COVID images. Further improvements can be made by tuning the model's architecture or using data augmentation techniques to enhance its performance.
