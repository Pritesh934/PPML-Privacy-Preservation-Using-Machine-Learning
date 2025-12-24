# 🔒 Privacy Preservation Using Machine Learning

This project focuses on **Privacy Preservation** in the domain of digital media by developing a robust **Image Forgery Detection System**. Leveraging **Generative Adversarial Networks (GANs)** and **Convolutional Neural Networks (CNNs)**, the system is designed to detect manipulated or tampered images (such as splicing or copy-move forgeries), thereby protecting user identity and ensuring data integrity in an era of Deepfakes.

## 📂 Project Overview

With the rise of facial recognition and advanced image editing tools, personal privacy is increasingly threatened by non-consensual image manipulation. This project addresses these challenges by:

1.  **Image Forensics**: Analyzing the **CASIA v2** dataset to distinguish between Authentic (`Au`) and Tampered (`Tp`) images.
2.  **Data Augmentation with GANs**: Implementing a Generative Adversarial Network (GAN) to generate synthetic image samples, addressing class imbalance and improving the robustness of the detector.
3.  **Forgery Detection**: Training a Deep Learning classifier (CNN) to accurately identify tampered images.

## 📊 Dataset

The project utilizes the **CASIA Image Tampering Detection Dataset v2.0**, a benchmark dataset for image forensics.

* **Source**: [Kaggle - CASIA Dataset](https://www.kaggle.com/sophatvathana/casia-dataset)
* **Structure**:
    * **Au (Authentic)**: Original, unaltered images.
    * **Tp (Tampered)**: Images manipulated via splicing or blurring techniques.
* **Preprocessing**: Images are resized, normalized, and converted to Error Level Analysis (ELA) maps (if applicable in your specific run) or raw pixel arrays for analysis.

## 🛠️ Technologies & Libraries Used

The project is implemented in Python using a Jupyter Notebook. Key technologies include:

* **Deep Learning Framework**: `TensorFlow` / `Keras`
* **Computer Vision**: `OpenCV` (`cv2`), `PIL` (Python Imaging Library)
* **Data Manipulation**: `NumPy`, `Pandas`
* **Visualization**: `Matplotlib`, `Seaborn`
* **Utilities**: `tqdm` (for progress bars), `opendatasets` (for Kaggle API integration)

## 🧠 Models Implemented

The solution involves a hybrid architecture:

1.  **Generative Adversarial Network (GAN)**:
    * **Generator**: Uses `Conv2DTranspose` layers to upsample noise into realistic image patches.
    * **Discriminator**: A CNN-based binary classifier that learns to distinguish between real dataset images and generated fakes.
    * **Purpose**: Used to augment the training data and enhance the model's ability to generalize against varied forgery attacks.

2.  **Convolutional Neural Network (CNN)**:
    * A standard supervised classifier trained on the augmented dataset to output a probability score for Image Authenticity.
    * Layers include `Conv2D`, `MaxPooling2D`, `BatchNormalization`, and `Dropout` for regularization.

## 🚀 How to Run

1.  **Clone this repository.**
2.  **Install Dependencies**:
    ```bash
    pip install tensorflow numpy matplotlib opencv-python opendatasets tqdm
    ```
3.  **Dataset Setup**:
    * The notebook is configured to download the dataset directly from Kaggle using `opendatasets`.
    * Ensure you have your `kaggle.json` API key ready.
    * The code looks for data in: `/content/casia-dataset/CASIA2/`
4.  **Run the Notebook**:
    ```bash
    jupyter notebook "PPML Project CASIA.ipynb"
    ```
5.  **Execution Flow**:
    * Run the initial cells to download and extract the data.
    * Execute the preprocessing blocks to load `Au` and `Tp` images.
    * Train the GAN to generate augmented samples.
    * Train the final Classifier to evaluate performance.

## Results (GAN)

- Data Augmentation: 1500 Synthetic Images Created
  
- Level of Improvement in Forgery Detection Accuracy: 1.50 %
