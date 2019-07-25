


#_______________________________________________________________________________________________
#   IMPORT CEREMONY

from __future__ import absolute_import, division, print_function

#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Print Version
print("TensorFlow " + tf.__version__) 

def skipLine(n):
    i=0
    if n <= 1:
        print("")
    else:
        for i in range(n-1):
            print("\n")

#   IMPORT MNIST FASHION DATASET
def importData():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    #   Label   Class
    #   0   	T-shirt/top
    #   1	    Trouser
    #   2   	Pullover
    #   3	    Dress
    #   4	    Coat
    #   5	    Sandal
    #   6	    Shirt
    #   7	    Sneaker
    #   8	    Bag
    #   9	    Ankle boot

    #   Each image is mapped to a single label. Since the class names are not included with the dataset, 
    #   store them here to use later when plotting the images:
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return train_images, train_labels, test_images, test_labels, class_names
#   Data Inspection and Pre-Processing
def imageInspection(i):
    skipLine(1)
    print("Display/Inspect 40000th picture.  pixel values fall in the range of 0 to 255")
    skipLine(1)

    plt.figure()
    plt.imshow(train_images[i])
    plt.colorbar()
    plt.grid(False)
    plt.show()
def imagePreProcess(train_images, test_images):
    #scale these values to a range of 0 to 1 before feeding to the neural network model
    #divide the values by 255. It's important that the training set and the testing set 
    #are preprocessed in the same way
    train_images = train_images / 255.0     #For the 60,000 training images (Control Images)
    test_images = test_images / 255.0       #For the 10,000 testing sample images
    return train_images, test_images;
#   Display several Inspected Images in a window
def displayFirstXImages():
    #Display first 25 from the Training set @ display the class name below each image
    #verify data is in correct format

    print("Display first 64 from the Training set display the class name below each image verify data is in correct format")
    skipLine(1)

    plt.figure(figsize=(8,8))
    for i in range(8*8):
        plt.subplot(8,8,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
#   Building the Neural Network for Learning
def neuralNetworkComponent(i):
    #Neural Network layers
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    #Compile the Model
    #Loss function — This measures how accurate the model is during training. 
    #               We want to minimize this function to "steer" the model in the right direction.
    #Optimizer — This is how the model is updated based on the data it sees and its loss function.
    #Metrics — Used to monitor the training and testing steps. The following example uses accuracy,
    #          the fraction of the images that are correctly classified.
    model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    #__________________________________________________________________________________________________
    #   TRAINING THE MODEL
    #   1   Feed the training data to the model—in this example, the train_images and train_labels arrays.
    #   2   The model learns to associate images and labels.
    #   3   We ask the model to make predictions about a test set—in this example, the test_images array. 
    #           We verify that the predictions match the labels from the test_labels array.
    model.fit(train_images, train_labels, epochs=i)
    return model;
#   Evaluating Accuracy of the Model
def evaluateAccuracy(test_images, test_labels, model):
    skipLine(1)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    skipLine(1)
#   Make predictions of the 10,000 images
def Prediction(n,model,test_images):
    print("A prediction is an array of 10 numbers. \nThese describe the confidence",
          " of the model that the image corresponds to each of the 10 different articles of clothing\n")
    predictions = model.predict(test_images)
    print(predictions[n])  #First Prediction
    #returns n'th highest confidence level in the prediction array returned above
    skipLine(1)
    print("{0}{1}".format("Highest Prediction Confidence in test_images (): ",np.argmax(predictions[n])))
    #Check whether prediction is correct by showing us the highest in test label (Answer)
    print("{0}{1}".format("Highest Prediction Confidence in test_labels : ",test_labels[n]))
    #Display name of Class with highest confidence.
    print("{0}{1}".format("Class Name with Highest Prediction Confidence : ",class_names[n-1]))
    return predictions;
#   Graph the Confidence of the Full Set of the Classes
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
#Display Individual Images i from the test_images set Predictions
def displayObjectPrediction(i):

    print("{0}{1}{2}".format("Displaying ", i, "th image prediction Results"))

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions,  test_labels)
    _ = plt.xticks(range(10), class_names, rotation=90)
    plt.show()
def plotSeveralImagePredictions(predictions,test_labels,test_images):
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, 2*num_cols, 2*i+1)
      plot_image(i, predictions, test_labels, test_images)
      plt.subplot(num_rows, 2*num_cols, 2*i+2)
      plot_value_array(i, predictions, test_labels)
    plt.show()



#   Data import
train_images, train_labels, test_images, test_labels, class_names = importData()
#Display/Inspect 40000th picture.  pixel values fall in the range of 0 to 255
imageInspection(40000)
#Pre-process images
train_images, test_images = imagePreProcess(train_images, test_images)
#The section Below shows the objects in a single window. TAKE NOTE.
displayFirstXImages()
#Training Model
model = neuralNetworkComponent(3)
#Evaluate Accuracy of the Model
evaluateAccuracy(test_images, test_labels, model)
#Predictions of test_images set
predictions = Prediction(0,model,test_images)
skipLine(1)
#Results for an image specified
displayObjectPrediction(0)
displayObjectPrediction(1210)
# Plot the Several test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
plotSeveralImagePredictions(predictions,test_labels,test_images)


i=1210
plt.figure(figsize=(6,3))
plt.subplot(1,3,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,3,2)
plot_value_array(i, predictions,  test_labels)

img = test_images[i]
print(img.shape)
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = model.predict(img)
print("prediction single")
print(predictions_single)
plt.subplot(1,3,3)
plot_value_array(0, predictions_single, test_labels) 
_ = plt.xticks(range(10), class_names, rotation=90)
np.argmax(predictions_single[0])
plt.show()
