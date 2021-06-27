import os

# Set Directory path for Dataset
os.chdir("C:/Users/erkam/PycharmProjects/pythonProject5/Tez")
Dataset='dataset'
Data_Dir=os.listdir(Dataset)
print(Data_Dir)
# Import necessary libraries
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

img_rows, img_cols = 112, 112

images = []
labels = []

for category in Data_Dir:
    folder_path = os.path.join('C:/Users/erkam/PycharmProjects/pythonProject5/Tez/Dataset', category)
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        img = cv2.imread(img_path)

        try:
            # Coverting the image into gray scale
            grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # resizing the gray scaled image into size 56x56 in order to keep size of the images consistent
            resized_img = cv2.resize(grayscale_img, (img_rows, img_cols))
            images.append(resized_img)
            labels.append(category)
        # Exception Handling in case any error occurs
        except Exception as e:
            print('Exception:', e)

images = np.array(images) / 255.0
images = np.reshape(images, (images.shape[0], img_rows, img_cols, 1))

# Perform one hot encoding on the labels since the label are in textual form
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
labels = np.array(labels)

(train_X, test_X, train_y, test_y) = train_test_split(images, labels, test_size=0.25,
                                                      random_state=0)
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D

# Define model paramters
num_classes = 2
batch_size = 32

# Build CNN model using Sequential API
model=Sequential()
#First layer group containing Convolution, Relu and MaxPooling layers
model.add(Conv2D(64,(3,3),input_shape=(img_rows, img_cols, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Second layer group containing Convolution, Relu and MaxPooling layers
model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Flatten and Dropout Layer to stack the output convolutions above
# as well as cater overfitting
model.add(Flatten())
model.add(Dropout(0.5))

# Softmax Classifier
model.add(Dense(64,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

print(model.summary())
from keras.optimizers import Adam

epochs = 950

model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics = ['accuracy'])

fitted_model = model.fit(
    train_X,
    train_y,
    epochs = epochs,
    validation_split=0.25)
## Plot the Training Loss & Accuracy

from matplotlib import pyplot as plt
# Plot Training and Validation Loss
plt.plot(fitted_model.history['loss'],'r',label='training loss')
plt.plot(fitted_model.history['val_loss'],label='validation loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss Value')
plt.legend()
plt.show()

# Plot Training and Validation Accuracy
plt.plot(fitted_model.history['accuracy'],'r',label='training accuracy')
plt.plot(fitted_model.history['val_accuracy'],label='validation accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy Value')
plt.legend()
plt.show()
# Save or Serialize the model with the name face_mask_detection_alert_system
model.save('face_mask_detection_alert_system.h5')