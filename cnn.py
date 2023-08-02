from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from os import listdir
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns
import pandas as pd

def f_cnn(_dir, _n):
    direc = _dir
    names = listdir(direc)
    train_images = []
    for i in range(0,_n):
        try:
            filename = direc + names[i]
            image = plt.imread(filename)
            image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image = cv2.resize(image, (30,30), interpolation = cv2.INTER_AREA)
            train_images.append(image)
        except:
            pass
    return train_images
        
img_b = f_cnn("/home/alex/Documents/melanoma_cancer_dataset/dataset/benign/", 500)
img_m = f_cnn("/home/alex/Documents/melanoma_cancer_dataset/dataset/malignant/", 500)

X = img_b + img_m
y = np.zeros((1000))
y[500:1000] = 1

train_images, test_images, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

input_shape = train_images[150].shape

model = Sequential()
model.add(Conv2D(30, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(2, activation=tf.nn.softmax))

train_images = np.array(train_images)
y_train = np.array(y_train)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=train_images,y=y_train, epochs=10)

test_images = np.array(test_images)
y_predict = model.predict(test_images)
y_predict = np.argmax(y_predict,axis=1)

print(y_predict)
print(y_test)

print('Precisión: ', precision_score(y_test, y_predict, average='binary'))
print('Recall: ', recall_score(y_test, y_predict, average='binary'))
print('F1 Score: ', f1_score(y_test, y_predict, average='binary'))
print('Accuracy: ', accuracy_score(y_test, y_predict))

def conf_matrix(y_true, y_predict):
    classes = np.arange(0,2)

    array = confusion_matrix(y_predict, y_true)
    df_cm = pd.DataFrame(array, index = classes,
                      columns = classes)
    plt.figure(figsize = (10,8))
    cmap = plt.get_cmap("YlOrRd", df_cm.values.max() - df_cm.values.min()+1)
    sns.heatmap(df_cm, annot=True, cmap=cmap, fmt='d', annot_kws={'clip_box':'tight'})
    plt.ylabel('True', fontsize=16)
    plt.xlabel('Predicted', fontsize=16)
    plt.tight_layout()
    plt.savefig('matriz_cnn.png')

conf_matrix(y_test, y_predict)
