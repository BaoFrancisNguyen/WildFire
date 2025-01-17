import streamlit as st
import zipfile
import os
import shutil
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# Déclaration globale du modèle
model = None
model_path = "saved_model.h5"

# Définition du modèle avec régularisation et images en 299x299
def build_model(num_classes):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(299, 299, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        GlobalAveragePooling2D(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.7),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Prétraitement des données avec augmentation améliorée
def data_generator():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# Extraction des fichiers ZIP sans inclure le dossier parent et écraser les fichiers existants
def extract_zip(zip_file, extract_to):
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to)
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        extracted_items = os.listdir(extract_to)
        for item in extracted_items:
            item_path = os.path.join(extract_to, item)
            if os.path.isdir(item_path):
                for sub_item in os.listdir(item_path):
                    src = os.path.join(item_path, sub_item)
                    dst = os.path.join(extract_to, sub_item)
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.move(src, extract_to)
                shutil.rmtree(item_path)
        st.write("📂 **Fichiers extraits :**", os.listdir(extract_to))

# Prétraitement d'une image pour la prédiction
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Interface Streamlit
st.title("Optimisation de Modèle de Classification d'Images")

# Paramètres d'entraînement
epochs = st.slider("Nombre d'époques", min_value=5, max_value=100, value=30, step=5)
learning_rate = st.number_input("Taux d'apprentissage", min_value=0.00001, max_value=0.01, value=0.0001, step=0.00001, format="%f")
optimizer_choice = st.selectbox("Choisir l'optimiseur", ["Adam", "SGD", "RMSprop"])

# Chargement des datasets
train_data = st.file_uploader("Importer les données d'entraînement (ZIP)", type=["zip"])

# Dossiers temporaires pour les données
train_dir = "temp_train_dir"

datagen = data_generator()

if train_data:
    if st.button("Démarrer l'entraînement"):
        extract_zip(train_data, train_dir)
        classes_detected = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        st.write(f"📂 **Classes détectées :** {classes_detected}")

        train_generator = datagen.flow_from_directory(
            train_dir, target_size=(299, 299), batch_size=32, class_mode='categorical', shuffle=True)

        num_classes = len(train_generator.class_indices)
        st.write(f"**Nombre de classes détectées : {num_classes}**")
        st.write("**Liste des classes détectées :**")
        st.write(train_generator.class_indices)

        model = build_model(num_classes)

        if optimizer_choice == "Adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_choice == "SGD":
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer_choice == "RMSprop":
            optimizer = RMSprop(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(train_generator, epochs=epochs, callbacks=[early_stop])
        model.save(model_path)  #Sauvegarde du modèle
        st.success("Entraînement terminé et modèle sauvegardé !")

#Prédiction sur une image
st.header("Faire une prédiction sur une image")
image_file = st.file_uploader("Choisissez une image pour prédiction", type=["jpg", "png"])

if image_file and st.button("Prédire"):
    if not os.path.exists(model_path):
        st.error("Le modèle n'est pas encore entraîné. Veuillez d'abord l'entraîner.")
    else:
        model = load_model(model_path)  #Chargement du modèle sauvegardé
        img_array = preprocess_image(image_file)
        prediction = model.predict(img_array)[0]
        predicted_class = np.argmax(prediction)
        confidence = round(prediction[predicted_class] * 100, 2)
        st.image(image_file, caption="Image à prédire")
        st.write(f"**Classe prédite : {predicted_class}** avec une confiance de **{confidence}%**")


















