## Satellite-Data-Processing-for-Detecting-Pollution-Sources

The dataset comprises 90 different animal images. Initially, we'll structure it for one-vs-rest classification, followed by binary classification and then a 5-class classification problem. We'll evaluate each model's performance using classification matrices.

1. **Dataset Preparation**:  
   - Organize the dataset for one-vs-rest classification. Perform binary classification using existing architectures and then restructure for 5-class classification. Use 3-fold cross-validation to assess the model.  
   - Preprocess images, including resizing, normalization, and data augmentation, to enhance model robustness and generalization.  
   - Split the dataset into training, validation, and test sets, ensuring balanced class distributions across splits.  

2. **Model Development**:  
   - Build a custom CNN model without using existing architectures like ResNet or DenseNet.  
   - Design the CNN architecture to include convolutional layers, pooling layers, and fully connected layers with ReLU activation and dropout regularization.  
   - Experiment with varying filter sizes, kernel dimensions, and layer depths to optimize feature extraction.  

3. **Training and Evaluation**:  
   - Train the model on prepared datasets for one-vs-rest and 5-class classification.  
   - Optimize hyperparameters such as learning rate, batch size, and number of epochs to improve performance.  
   - Generate classification matrices for visualization and compare precision, recall, and F1 scores across classes.  
   - Track the training and validation loss over epochs to monitor overfitting and underfitting.  
   - Use metrics like accuracy, confusion matrices, and AUC-ROC curves to comprehensively evaluate performance.  

4. **Convolutional Layer Visualization**:  
   - Plot the output of all convolutional layers and discuss the insights on automatically created features.  
   - Analyze feature maps to understand which patterns the model focuses on for each class, providing interpretability for the classification decisions.  
   - Visualize filters and gradients to identify the layers that contribute most to model performance and highlight potential areas for optimization.
