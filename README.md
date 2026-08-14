# Brain Tumor Detection

## Introduction

### Problem Statement

Brain tumors are **critical medical conditions** that require prompt and accurate diagnosis for effective treatment. **Traditional methods** of detecting brain tumors through manual examination of MRI scans by radiologists are often **time-consuming, prone to human error, and can delay necessary treatment**. As the volume of medical imaging data continues to grow, there is a **significant need for automated systems** to assist in **early detection** and diagnosis.

<div align="center">
    <table>
        <tr>
            <td align="center">
                <img src="https://github.com/user-attachments/assets/ca389bf9-fa8d-4ab7-941c-91950397db9a" alt="Tumor Brain" width="300" height="300"/>
                <p><b>Tumor Brain</b></p>
            </td>
            <td align="center">
                <img src="https://github.com/user-attachments/assets/b3a41b08-7f53-4dfa-a161-78b60ac35b2a" alt="Healthy Brain" width="300"height="300"/>
                <p><b>Healthy Brain</b></p>
            </td>
        </tr>
    </table>
</div>

### Project Overview

This project aims to develop an automated brain tumor detection system using **Convolutional Neural Networks (CNNs)**, a type of deep learning model particularly effective for image analysis tasks due to their ability to learn spatial hierarchies of features. The CNN architecture is designed to automatically and adaptively learn spatial hierarchies of features through **backpropagation** by utilizing multiple building blocks, such as **convolutional layers, pooling layers, and fully connected layers**. The model starts by applying various **convolutional filters** to the input MRI images to **extract low-level features** like **edges and textures**, then progresses to **higher-level features** that may represent more **complex structures** such as tumors.

The dataset used in this project consists of **brain MRI images** that have been carefully labeled as either containing a **tumor or healthy**. These images undergo a series of **preprocessing steps**, including **resizing, normalization, and data augmentation**, to enhance the quality and variability of the training data, which helps the model generalize better to new, unseen images. During training, the CNN adjusts its filters and weights through numerous iterations to minimize the classification error, learning to distinguish between subtle differences in MRI scans that **indicate the presence of a tumor**.

Additionally, advanced techniques such as dropout and **batch normalization** are employed to **prevent overfitting** and ensure robust learning. The model's performance is evaluated using metrics such as accuracy and validation accuracy on the test set to ensure it can reliably identify tumors in diverse clinical scenarios. This deep learning-based approach **leverages the power of CNNs to process and analyze medical images efficiently**, enhancing diagnostic accuracy and significantly reducing the time required for analysis, ultimately improving patient outcomes by providing timely and reliable assistance to radiologists.

## Data Preprocessing

Data preprocessing is a crucial step in preparing the MRI images for training the Convolutional Neural Network (CNN). Proper preprocessing ensures the model is trained on well-prepared data, enhancing its ability to generalize to new, unseen data. The table below outlines the preprocessing methods applied to the brain MRI images:

| **Step**                      | **Description**                                                                                                                                                                                                                                                     |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data Collection and Labeling** | The dataset consists of MRI images of the brain, labeled into two categories: "tumor" and "healthy." These labels are essential for supervised learning, where the model learns to classify images based on the labeled data.                                                   |
| **Image Resizing**            | All images were resized to a uniform dimension, which is necessary for feeding images into the neural network. This step helps reduce computational requirements and speeds up the training process. Images were resized to a standard size using TensorFlow’s preprocessing tools.              |
| **Normalization**             | The pixel values of the MRI images were normalized to a range of 0 to 1 using TensorFlow’s preprocessing capabilities. Normalization stabilizes the learning process and improves convergence during training, ensuring efficient learning without bias from varying intensity scales of the raw images. |
| **Data Augmentation**         | Data augmentation techniques were applied using `ImageDataGenerator` from TensorFlow's Keras API. Methods included rotation, flipping, zooming, and shifting to increase dataset diversity and prevent overfitting, simulating real-world variations in medical imaging.                                   |
| **Preprocessing the Training Set** | The training set underwent extensive preprocessing to enhance learning, including rotation, zooming, flipping, and shifting. These augmentations simulate different angles, scales, orientations, and spatial variances to make the model more robust and generalize better to new images.                  |
| **Preprocessing the Test Set**     | The test set was preprocessed to maintain consistency with the training data. Augmentation was not applied to ensure a fair evaluation, but normalization and resizing were performed to maintain uniformity across the dataset.                                                  |
| **Splitting the Dataset**     | The dataset was split into training, validation, and testing sets, typically at a 70:15:15 ratio. This approach ensures the model is trained on substantial data while being evaluated on unseen images to gauge performance.                                                |
| **Image Batching and Shuffling**   | Images were batched and shuffled during training to ensure the model learns from a diverse set in each epoch. Batching groups a fixed number of images into batches for simultaneous model processing, speeding up training, while shuffling prevents learning patterns based on data order.               |

These preprocessing steps are essential for preparing the MRI images for training a robust and accurate CNN model. By carefully preprocessing the data, the model is better equipped to learn meaningful patterns from the images, leading to improved performance in detecting brain tumors.

## Convolutional Neural Network (CNN)

### Introduction to CNN Architecture

In this brain tumor detection project, a Convolutional Neural Network (CNN) is employed to classify MRI images into two categories: "tumor" and "healthy." CNNs are highly effective for image recognition tasks because they automatically learn to detect patterns and features in images through layers of convolutions and activations. The architecture is specifically designed to extract relevant features from MRI scans, enhancing the model's ability to differentiate between tumor and non-tumor cases.

### Detailed CNN Architecture and Steps

1. **Input Layer:**
   - The input layer receives the preprocessed MRI images, resized to a standard dimension of **128x128x3** pixels. This uniform size ensures that all images are compatible with the CNN architecture and provides a consistent input for the model to process.

2. **First Convolutional Layer:**
   - The first convolutional layer applies **32 filters** of size **3x3** to the input images. This layer is responsible for detecting low-level features such as edges and textures. A stride of **1** and 'same' padding are used to maintain the spatial dimensions of the input. This layer captures the initial features that are crucial for identifying patterns in MRI scans.

3. **Activation Function (ReLU):**
   - Following the first convolutional layer, the **ReLU (Rectified Linear Unit)** activation function is applied. ReLU introduces non-linearity into the model by replacing all negative pixel values in the feature map with zero, allowing the network to learn more complex patterns.

4. **First Pooling Layer:**
   - A **Max Pooling** layer with a **2x2** pool size is used to downsample the feature maps. This reduces the spatial dimensions by half, decreasing the computational complexity and helping the model focus on the most prominent features in each region of the image.

5. **Second Convolutional Layer:**
   - The second convolutional layer applies **64 filters** of size **3x3** to the output of the first pooling layer. This layer enables the model to learn more complex features by building on the initial features extracted in the first convolutional layer. The stride and padding remain consistent with the previous layer to further refine the model’s understanding of the input images.

6. **Activation Function (ReLU):**
   - Another **ReLU** activation function is applied to the output of the second convolutional layer, allowing the model to capture more intricate patterns in the data.

7. **Second Pooling Layer:**
   - Another **Max Pooling** layer with a **2x2** pool size is used to further reduce the spatial dimensions of the feature maps. This pooling layer continues to decrease the computational requirements and helps the model focus on the most significant features.

8. **Flatten Layer:**
   - The **Flatten** layer converts the 2D feature maps into a 1D vector, which can then be fed into fully connected layers. This transformation is essential for transitioning from the convolutional layers to the dense layers that perform the final classification.

9. **Fully Connected Layer (Dense):**
   - A fully connected layer with **128 neurons** is added. Each neuron in this layer is connected to every neuron in the previous layer, allowing the model to learn complex combinations of features that contribute to the final classification. A **Dropout** layer is applied here with a rate of **0.5** to prevent overfitting by randomly setting a fraction of input units to zero during training.

10. **Output Layer:**
    - The output layer is a fully connected layer with **2 neurons**, corresponding to the two classes: "tumor" and "healthy." This layer uses a **softmax** activation function to provide a probability distribution over the two classes, enabling the model to predict the likelihood of the input image belonging to each class.

11. **Loss Function and Optimization:**
    - The model uses the **categorical cross-entropy** loss function to measure the difference between the predicted class probabilities and the actual class labels. The **Adam optimizer** is employed to adjust the model's weights based on the gradients of the loss function, minimizing the error and improving accuracy over time.

12. **Training the Model:**
    - The model is trained using the training dataset over a specified number of **epochs** (e.g., 50 epochs). During each epoch, the model iterates over the entire dataset, adjusting the weights and biases to minimize the loss. The training process is monitored using validation data to ensure the model is learning effectively and not overfitting.

These steps outline the specific architecture and reasoning behind building and training the CNN model for brain tumor detection. Each component of the model plays a critical role in ensuring it learns to distinguish between tumor and healthy images accurately, ultimately aiding in the early detection and diagnosis of brain tumors.


| **Layer**                     | **Details**                                                                                     |
|-------------------------------|------------------------------------------------------------------------------------------------|
| Input Layer               | Receives preprocessed MRI images resized to **128x128x3** pixels.                              |
| First Convolutional Layer | Applies **32 filters** of size **3x3** to extract low-level features.                          |
| ReLU Activation            | Introduces non-linearity by replacing negative pixel values with zero.                         |
| First Pooling Layer       | **Max Pooling** with **2x2** pool size to downsample feature maps and reduce dimensions.       |
| Second Convolutional Layer| Applies **64 filters** of size **3x3** to learn more complex features.                         |
| ReLU Activation           | Further non-linearity after the second convolutional layer.                                    |
| Second Pooling Layer      | **Max Pooling** with **2x2** pool size to reduce dimensions further.                           |
| Flatten Layer             | Converts 2D feature maps into a 1D vector for dense layers.                                    |
| Fully Connected Layer     | **128 neurons** with **Dropout** rate of **0.5** to prevent overfitting.                       |
| Output Layer              | **2 neurons** with **softmax** activation for class probability distribution.                  |
| Loss Function             | Uses **categorical cross-entropy** to measure prediction accuracy.                             |
| Optimizer                 | **Adam optimizer** to adjust model weights and minimize loss.                                  |
| Training                  | Trained over **50 epochs** with monitoring using validation data to avoid overfitting.         |
| Evaluation                | Evaluated on test data using **accuracy, precision, recall,** and **F1-score**.                |


## Model Evaluation and Results

After training the Convolutional Neural Network (CNN) model for brain tumor detection, it was crucial to evaluate its performance on both training and validation datasets. The metrics used for evaluation were accuracy and loss, which provide insights into how well the model has learned to distinguish between tumor and healthy brain MRI images. The results indicate the model's effectiveness and its ability to generalize to new, unseen data.

### Final Results

After training the CNN model for brain tumor detection for 25 epochs, the following final evaluation metrics were recorded:

| **Metric**           | **Final Value (Training Set)** | **Final Value (Validation Set)** |
|----------------------|-------------------------------|----------------------------------|
| **Accuracy**         | 0.9795                        | 0.9108                           |
| **Loss**             | 0.0716                        | 0.2556                           |

- **Training Accuracy:** The model achieved a high accuracy of **97.95%** on the training set, indicating that it effectively learned the patterns in the training data.
- **Validation Accuracy:** The model achieved an accuracy of **91.08%** on the validation set, suggesting good generalization to new, unseen data without significant overfitting.
- **Training Loss:** The final training loss was **0.0716**, showing that the model effectively minimized the difference between predicted and actual labels during training.
- **Validation Loss:** The validation loss was **0.2556**, which stabilized, indicating that the model reached an optimal point without overfitting.

These results demonstrate the CNN model's effectiveness in accurately detecting brain tumors from MRI images, making it a promising tool for clinical diagnosis.
