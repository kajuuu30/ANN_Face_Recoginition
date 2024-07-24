# Face Recognition System using PCA and ANN

This project implements a face recognition system using Principal Component Analysis (PCA) and Artificial Neural Networks (ANN) in Python.

## Libraries Used

- NumPy
- SciPy
- OpenCV-Python

## Implementation Steps
1. Data loading and preprocessing
2. PCA for feature extraction and dimensionality reduction
3. LDA for further feature extraction
4. MLP classifier training
5. Model evaluation and visualization of results
   
### Training

1. Generate face database
2. Calculate mean face
3. Perform mean subtraction
4. Calculate covariance matrix
5. Perform eigenvalue and eigenvector decomposition
6. Select best eigenvectors (feature vectors)
7. Generate eigenfaces
8. Create face signatures
9. Train ANN using backpropagation

### Testing

1. Preprocess test image
2. Perform mean subtraction
3. Project onto eigenfaces
4. Use trained ANN model for prediction

## Evaluation

- 60% of data used for training, 40% for testing
- Accuracy evaluated for different values of k (number of eigenvectors)
- Graph plotted to show relationship between k and classification accuracy
- Imposters (non-enrolled individuals) added to test set to evaluate system robustness

## Future Work

- Experiment with different ANN architectures
- Implement additional face recognition algorithms for comparison
- Optimize for real-time performance

## References

1. Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. Journal of cognitive neuroscience, 3(1), 71-86.

## Features
- Principal Component Analysis (PCA) for dimensionality reduction
- Linear Discriminant Analysis (LDA) for further feature extraction
- Multi-Layer Perceptron (MLP) classifier for face recognition
- Visualization of eigenfaces



## Results
![Eigenfaces](path_to_eigenfaces_image.png)
![Figure_1](https://github.com/user-attachments/assets/d5d7a55c-b72e-4bea-8635-5558369b78f6)
![image](https://github.com/user-attachments/assets/aa0a141b-29d7-4ee0-9f4b-8dd2c3b1aece)



The image above shows the top 12 eigenfaces extracted from the dataset. These eigenfaces represent the principal components of variation in the facial images.

Accuracy: [Insert accuracy here after running the code]

The eigenfaces demonstrate how the PCA algorithm captures the most significant features of faces. The first eigenface (top left) typically represents overall face shape and lighting, while subsequent eigenfaces capture more specific facial features and variations.

## Usage
1. Prepare your dataset
2. Update the `dir_name` variable in the code to point to your dataset.
3. Run the script to train the model and visualize results.

## Parameters
- Number of PCA components: 150
- MLP architecture: 2 hidden layers with 10 neurons each
- Train-test split: 75% training, 25% testing
