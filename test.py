from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm  # pour viisualiser le progres
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications import VGG16
from keras.models import Model



# Preprocesser les images
def preprocess_images(folder_path, target_size=(128, 128)):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            try:
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(folder_path.split('/')[-1])  # Extract label from folder name
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    return np.array(images), labels

# Preprocess parasitized images
parasitized_images, parasitized_labels = preprocess_images(parasitized_path)

# Preprocess uninfected images
uninfected_images, uninfected_labels = preprocess_images(uninfected_path)

# Check and ensure all images have the same shape
print("Parasitized Image Shapes:", set(img.shape for img in parasitized_images))
print("Uninfected Image Shapes:", set(img.shape for img in uninfected_images))

# Concatenate the preprocessed data
images = np.concatenate((parasitized_images, uninfected_images))
labels = parasitized_labels + uninfected_labels

# Normalisation des images
images_normalized = images / 255.0

# Encodage des labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Sanity check
print("Shape of Images:", images.shape)
print("Length of Labels:", len(labels_encoded))


# Augmentation des données
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)



# Affichage d'exemples d'images augmentées
# Choisir une image à augmenter
img = images_normalized[0]
img = img.reshape((1,) + img.shape)

# Afficher quelques exemples d'images augmentées
plt.figure(figsize=(10, 10))
i = 0
for batch in datagen.flow(img, batch_size=1):
    plt.subplot(3, 3, i + 1)
    plt.imshow(batch[0])
    i += 1
    if i % 9 == 0:
        break
plt.show()



# Determine the dimensions (width and height) of your preprocessed images
img_height, img_width, channels = images.shape[1:]  # Assuming 'images' is your preprocessed image data

# Determine the number of unique classes from the encoded labels
num_classes = len(np.unique(labels_encoded))

# Define your model with the correct input shape and number of classes
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])



base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Récupération de l'encodeur pré-entraîné
for layer in base_model.layers:
    layer.trainable = False

# Ajout des couches Dense
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model_vgg = Model(inputs=base_model.input, outputs=predictions)





# Définition des callbacks
callbacks = [
    EarlyStopping(patience=3, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.1, patience=2, monitor='val_loss')
]

# Compilation et entraînement des modèles
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(images_normalized, labels_encoded, epochs=10, validation_split=0.2, callbacks=callbacks)

# Sauvegarde des poids des modèles
model.save_weights('model_weights.weights.h5')