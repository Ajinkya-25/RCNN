import tensorflow as tf 
from tensorflow import GradientTape 
import numpy as np

num_anchors=1
target_size=(64,64)
classes=2

def regionnetwork(input_shape=(None, None, 3), num_anchors=9):
    model=tf.keras.models.Sequential([
        #tf.keras.layers.Input(input_shape),
        tf.keras.layers.Conv2D(128,(3,3),activation='relu', padding='same'),#510*510*256
        tf.keras.layers.MaxPool2D((3,3),padding='same'),#170*170*85
        #tf.keras.layers.Conv2D(128,(2,2),activation='linear'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(4,activation='sigmoid')#add 1 neuron for no of boxes if needed
    ])
    return model

def cnnnetwork():
    model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(2,2),padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D((3,3),padding='same'),
        tf.keras.layers.Conv2D(32,(2,2),padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D((2,2),padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'), 
        tf.keras.layers.Dense(classes, activation='softmax') 
    ])
    return model

class roi():
    def __init__(self) -> None:
        self.classifier=cnnnetwork()
        self.rgn=regionnetwork()
    
    def crop_and_resize(self,input_image,proposals):
        x1,y1,x2,y2=map(int,proposals)*255
        roi=tf.image.crop_to_bounding_box(input_image,y1, x1, y2 - y1, x2 - x1)
        resized_roi = tf.image.resize(roi, target_size)
        return resized_roi
    
    def prediction(self,input_image):
        proposals=self.rgn.predict(input_image)
        rois=self.crop_and_resize(input_image,proposals)
        classification=[self.classifier.predict(roi) for roi in rois]
        return classification
    
    def train(self,dataset,bbox=(64,64,64,64)):
        self.train_accuracy = tf.keras.metrics.Accuracy()
        epochs = 10
        for epoch in range(epochs):
            for input_images, names in dataset:
                with tf.GradientTape() as gt_classifier, tf.GradientTape() as gt_region:
                
                    pred_names = self.classifier(input_images, training=True)
                    classifier_loss = tf.keras.losses.sparse_categorical_crossentropy(names, pred_names)

                
                    region_output = self.rgn(input_images, training=True)
                    #print(region_output)
                    region_loss =tf.keras.losses.mae(bbox,region_output)

                self.train_accuracy.update_state(names, tf.argmax(pred_names, axis=1))
                optimizer_classifier = tf.keras.optimizers.Adam(learning_rate=0.01)
                optimizer_region = tf.keras.optimizers.Adam(learning_rate=0.01)

                gradients_classifier = gt_classifier.gradient(classifier_loss, self.classifier.trainable_variables)
                optimizer_classifier.apply_gradients(zip(gradients_classifier, self.classifier.trainable_variables))

                gradients_region = gt_region.gradient(region_loss, self.rgn.trainable_variables)
                optimizer_region.apply_gradients(zip(gradients_region, self.rgn.trainable_variables))
            epoch_accuracy = self.train_accuracy.result()
            print(f"Epoch {epoch + 1}, Classifier Loss: {classifier_loss.numpy()}, Region Loss: {region_loss.numpy()}, Accuracy: {epoch_accuracy.numpy()}")
            self.train_accuracy.reset_states()

rcnn=roi()

data_train = tf.keras.preprocessing.image_dataset_from_directory(
    "F:\cnn models\data123",
    validation_split=0.2,
    subset="training",
    shuffle=True,
    seed=42,
    image_size=(256, 256),
    batch_size=32,
)

data_test = tf.keras.preprocessing.image_dataset_from_directory(
    "F:\cnn models\data123",
    validation_split=0.2,
    subset="validation",
    shuffle=True,
    seed=42,
    image_size=(256, 256),
    batch_size=32,
)

data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)
data_test = data_test.prefetch(tf.data.experimental.AUTOTUNE)

input_images,classes=next(iter(data_train))
rcnn.train(data_train)

