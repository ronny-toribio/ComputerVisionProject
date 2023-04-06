# Computer Vision Project
This is a group project for our machine learning class using a convolution neural network model to predict emotions. Our group consists of [Kadir Altunel](https://github.com/KadirOrcunAltunel), [Michael Cook-Stahl](https://github.com/MichaelCook01) and myself [Ronny Toribio](https://github.com/ronny-phoenix). I obtained the facial emotional recognition data for 2013 from [FER2013](https://www.kaggle.com/code/ritikjain00/model-training-fer-13/data). I also looked at two candidates for the facial expression recognition challenge that use this dataset, [Candidate 1](https://www.kaggle.com/code/ritikjain00/model-training-fer-13/notebook) and [Candidate 2](https://www.kaggle.com/code/gauravsharma99/facial-emotion-recognition/notebook). In our machine learning class we had learned about SciKit-learn, Tensorflow and Keras for machine learning in Python. This project was an opportunity for us to explore using convolutional neural networks (CNN) for classication. We used Jupyter notebooks for most of this project.

# Data Wrangling
The first step in this project was to load the images using Tensorflow's ImageDataGenerator and split the images into a training set, a validation set and a test set. The validation set is 10% of the original training set. There are 7 classes of images corresponding to 7 emotions.

##### Datasets
- Training set contains 25841 images
- Validation set contains 2868 images
- Testing set contains 7178 images

# Our CNN model
For our model we used the Keras API to build a sequential model. We tried different numbers of convolution blocks eventually settling on 4 convolution blocks followed by a classification block. The convolution blocks contain two convolution layers, batch normalization, a max pooling layer, and a dropout layer set to randomly drop 25% of the output of the block to avoid overfitting. The functions that build our model have the option to use only three convolution blocks as a hyperparameter. The choice of activation functions used for the convolutional blocks was also a hyperparameter, as well as the corresponding initializers.

The classification block consists of two densely connected layers, with batch normalization and dropout before the final dense layer. The activation function of the first layer is the same as for the convolution layers and is a hyperparameter. The final layer's activation function is always softmax. Our model's optimizer is Adam, the loss function is categorial cross entropy and the metric used is accuracy.

##### Our model's structure
- Convolution block 0
   - Conv2D 32 (3, 3)
   - Conv2D 64 (3, 3)
   - BatchNormalization
   - MaxPooling2D (2, 2)
   - Dropout 25%
- Convolution block 1
   - Conv2D 128 (3, 3)
   - Conv2D 256 (3, 3)
   - BatchNormalization
   - MaxPooling2D (2, 2)
   - Dropout 25%
- Convolution block 2
   - Conv2D 512 (3, 3)
   - Conv2D 1024 (3, 3)
   - BatchNormalization
   - MaxPooling2D (2, 2)
   - Dropout 25%
- Convolution block 3 (Optional)
   - Conv2D 512 (3, 3)
   - Conv2D 1024 (3, 3)
   - BatchNormalization
   - MaxPooling2D (2, 2)
   - Dropout 25%
- Classification block
   - Flatten
   - Dense 2048
   - BatchNormalization
   - Dropout 50%
   - Dense 7
   
# Grid search of hyperparameters
Instead of training just one model we decided to apply what we learned about SciKit-Learn's GridSearch capability to try many different versions of our model and the version with the best accuracy. This proved difficult because SciKit-Learn didn't have a way to treat a Keras/Tensorflow model as a SciKit-Learn estimator for classification. I tried other libraries but ultimately ended up looking at the source code for SciKit-Learn's GridSearch and found a class named ParameterGrid that is used to iterate over hyperparameters. The hyperparamers we wanted to try included the choice of activation function, initializer and whether to use the fourth convolution block. Kadir used his Google Colab account to run our Jupyter notebook [train.ipynb](/train.ipynb) and generate the models and data.

##### Hyperparameters
```Python
param_grid = [
    {"main_activation": ["relu", "elu"], "main_initializer":["he_normal"], "use_conv_block3": [True, False]},
    {"main_activation": ["selu"], "main_initializer":["lecun_normal"], "use_conv_block3": [True, False]},
    {"main_activation_layer": [LeakyReLU], "main_activation_name": ["leaky_relu"],
     "main_initializer":["he_normal"], "use_conv_block3": [True, False]}
]
```

##### Callbacks
We also used four training callbacks. Early stopping and reduce learning rate on plateau to help with training. The csv logger to record the metrics at each epoch for every model candidate. We also used the checkpoint callback to save the best version of each candidate. We saved every candidate as well and keep track of the current best model throughout the grid search.

```Python
early_stopping_cb = EarlyStopping(min_delta=0.00005, patience=11, verbose=1, restore_best_weights=True)

reduce_lr_cb = ReduceLROnPlateau(factor=0.5, patience=7, min_l=1e-7, verbose=1)

csv_cb = CSVLogger(cur_name + "-training.csv")

checkpoint_cb = ModelCheckpoint(
    cur_weights,
    monitor = 'accuracy',
    verbose = 1, 
    save_best_only = True,
    save_weights_only = True
)
```

##### Results

```Python
Name: relu+he_normal+3-convolution-blocks Accuracy: 0.6802730560302734 Loss: 1.0993871688842773
Name: relu+he_normal+4-convolution-blocks Accuracy: 0.6765115857124329 Loss: 1.0885995626449585
Name: leaky_relu+he_normal+4-convolution-blocks Accuracy: 0.6652270555496216 Loss: 1.05915367603302
Name: leaky_relu+he_normal+3-convolution-blocks Accuracy: 0.6634159684181213 Loss: 1.0441060066223145
Name: elu+he_normal+3-convolution-blocks Accuracy: 0.6528280973434448 Loss: 1.0537306070327759
Name: selu+lecun_normal+4-convolution-blocks Accuracy: 0.6447478532791138 Loss: 1.1209967136383057
Name: selu+lecun_normal+3-convolution-blocks Accuracy: 0.6354137659072876 Loss: 1.1144425868988037
```

The best version of our model was ReLU for the activation function with he_normal as the initializer and three convolution blocks. The highest accuracy was 68% while the least accurate was 63%. The activation functions in descending order of accuracy for our grid search were ReLU, Leaky ReLU, ELU and SELU. Generally having four convolution blocks was better than having three, except for the winning candidate with three blocks.

# Testing
To test our model I wrote a Python script [analyze.py](/analyze.py) that loads our winning candidate model and an mp4 video. Using opencv I iterate through the video frame by frame. For each frame I use another CNN model to detect faces and isolate faces then use our model to predict the emotion on that face. We've chosen different colors for each of the seven emotions. With that color I draw a bounding box around the face and the name of that emotion above the bounding box.

##### Emotion colors
- Red (Anger)
- Green (Disgust)
- Gray (Fear)
- Yellow (Happy)
- Blue (Sad)
- Orange (Surprise)

##### Usage
```bash
analyze.py [model_name] [video.mp4]
```
The program expects a JSON file containing the Keras structure named model_name.json and the models weights named model_name.h5.

# Conclusion
Overall the project was a success. We were able to build several versions of our CNN model and select the most accurate. We were even able to use it to detect emotions from faces. Michael created a powerpoint presentation for this project [here](/machine_learning_presentation.pptx). After successfully completing this project, I tried to remove the dependency of my [analyze.py](/analyze.py) of using another model for facial detection before predicting emotions. I attempted to attach a Keras RetinaNet to our model in order to generate bounding boxes along with emotion classification. This attempt is [train2.ipynb](/train2.ipynb) and was ultimately unsuccessful. In the future I would like to revisit the bounding box regression.

# Authors
- [Ronny Toribio](https://github.com/ronny-toribio) - Project lead, Data Wrangling, Model design, Video application
- [Kadir Altunel](https://github.com/KadirOrcunAltunel) - Model design, Google Colab Training
- [Michael Cook-Stahl](https://github.com/MichaelCook01) - PowerPoint presentation

