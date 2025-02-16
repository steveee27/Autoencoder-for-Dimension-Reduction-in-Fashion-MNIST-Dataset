# Autoencoder for Dimension Reduction in Fashion MNIST Dataset

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results and Discussion](#results-and-discussion)
  - [Model Performance](#model-performance)
  - [Key Insights](#key-insights)
  - [Limitations](#limitations)
- [Conclusion](#conclusion)
- [License](#license)

## Project Overview

This project explores the use of an **Autoencoder** for dimension reduction using the **Fashion MNIST** dataset. The primary task is to reduce the 784-dimensional images (28x28) into 128-dimensional latent space while maintaining as much of the original image's information as possible. The model's ability to reconstruct the images is assessed using the **Structural Similarity Index (SSIM)**, which compares the reconstructed image to the original. The model was further optimized through modifications like the addition of convolution layers, dropout, and batch normalization.

## Dataset Overview

The dataset used for this project is **Fashion MNIST**, which consists of grayscale images of clothing items, including:

- **T-shirt/top**
- **Trouser**

These categories are the focus of the project. The dataset can be accessed [here](https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md).

The images are 28x28 pixels, and the goal is to reduce their dimensionality to 128, retaining key features for accurate reconstruction.

## Data Preprocessing

Before training the Autoencoder, the following steps were taken:

1. **Loading and Scaling**: The images were scaled to values between 0 and 1 to standardize the input for the Autoencoder.
2. **Splitting the Data**: The dataset was split into training, validation, and test sets with the following proportions:
   - **80%** for training
   - **10%** for validation
   - **10%** for testing
3. **Image Reshaping**: Each image was reshaped to include one channel (28x28x1) as Autoencoders expect images with channel dimensions.

## Model Architecture

The baseline model follows the **Autoencoder** architecture as shown below, with an encoder that compresses the input into a latent representation and a decoder that reconstructs the input.

### Autoencoder Architecture:
- **Encoder**:
  - Convolution layer with 32 filters, kernel size 3x3, ReLU activation.
  - MaxPooling layer with pool size 2x2.
  - Fully connected layer to reduce the dimensionality to 128.
- **Decoder**:
  - Fully connected layer to expand the latent space to a larger size.
  - Reshaping and upsampling to reconstruct the image.
  - Convolution layers for reconstructing the image, ending with a sigmoid activation function.

![image](https://github.com/user-attachments/assets/5fa24328-425a-4a80-ac6b-c98ce69f8d2f)


## Model Training and Evaluation

### Training:
- The model was trained for **50 epochs** using the **Adam optimizer** and **binary cross-entropy** loss function.
- Early stopping was applied to prevent overfitting, monitored on the validation loss.

### Evaluation:
The model performance was evaluated using the **Structural Similarity Index (SSIM)**. SSIM measures the visual similarity between the original and reconstructed images, where a higher SSIM indicates better reconstruction.

## Results and Discussion

### Model Performance:
- **Baseline Model (without modifications)**: The autoencoder was able to learn a latent representation of the input images and reconstruct them with a relatively high SSIM value of **0.913**.
- **Modified Architecture**: After implementing modifications such as additional convolution layers, batch normalization, dropout, and a higher latent dimension (256), the SSIM improved to **0.928**. 

### Key Insights:
1. **Effectiveness of Modifications**: The improvements in SSIM from **0.913** to **0.928** after architectural changes show that adding layers and optimizing hyperparameters significantly enhanced the model's ability to reconstruct images.
2. **Latent Space Representation**: Increasing the latent space dimensionality helped capture more features, improving image quality.
3. **Training Stability**: Batch normalization and dropout helped stabilize training and reduced overfitting, ensuring better generalization to unseen data.

### Limitations:
- **Limited Dataset Size**: Using Fashion MNIST, which has simple images, might not show the full potential of the architecture. The model may need to be tested on more complex datasets.
- **Overfitting Potential**: Despite using early stopping, the model's performance could still be affected by overfitting in cases where the latent space is too large.

## Conclusion

This project successfully demonstrated the use of an **Autoencoder** for **dimension reduction** of **Fashion MNIST images**. The **baseline Autoencoder** performed well with an SSIM of **0.913**, and after modifying the architecture, the model's performance improved to **0.928**, showcasing the effectiveness of architectural modifications. This method of dimension reduction can be adapted for various tasks, including image compression and feature extraction.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
