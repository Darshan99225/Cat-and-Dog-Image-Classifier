# Cat-and-Dog-Image-Classifier

# Setup and Data Preparation

Google Colaboratory: Use this platform for the project. Create a copy of the provided notebook.
Data Preparation: Download the dataset and set key variables. Create image generators for training, validation, and test datasets using ImageDataGenerator with rescaling.

# Model Building
Neural Network: Construct a convolutional neural network (CNN) using TensorFlow 2.0 and Keras. Use a stack of Conv2D and MaxPooling2D layers, followed by a fully connected layer with ReLU activation.
Compilation: Compile the model with an optimizer and loss function. Include metrics=['accuracy'] to monitor training and validation accuracy.

# Training the Model
Fit Method: Train the model using the fit method. Specify the training data, steps per epoch, number of epochs, validation data, and validation steps.
Data Augmentation: Use data augmentation techniques with ImageDataGenerator to avoid overfitting by applying random transformations to the training images.

# Evaluation and Visualization
Accuracy and Loss Visualization: Plot the accuracy and loss of the model to evaluate its performance.

Prediction on Test Data: Use the trained model to predict the class (cat or dog) of new test images. Visualize the test images with predicted probabilities.
