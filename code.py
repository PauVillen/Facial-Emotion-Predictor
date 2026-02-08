#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJECTE CPIA: Classificació d’emocions facials amb Xarxa Neuronal
Convolucional

Albert Ramos Nadal, Pau Villén Vidal
Desembre 2024
"""

#Importem directoris necessaris:
import os  
import numpy as np
import pandas as pd
import tensorflow as tf  #Per al model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization  #Capes per al model
from tensorflow.keras.optimizers import Adam   #Per compilar el model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  #Generador d'imatges
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import seaborn as sns  


num_classes = 6  #Número de classes (sense la categoria 'disgust')
batch_size = 64  #Adequat per a un model de classificació
epochs = 20  #Número que ens permet obtenir un bon model

#Funció que compta el nombre d'imatges que hi ha en cadascuna de les categories. 
def count_images(dir,name): 
  counts={}

  for emotion in os.listdir(dir):
    emotion_path=os.path.join(dir,emotion)
    if os.path.isdir(emotion_path):
      counts[emotion]=len(os.listdir(emotion_path))
  
  data_frame=pd.DataFrame(counts,index=[name])
  return data_frame  #Retorna un data frame que indica aquest nombre d'imatges per categoria per al train i test.

#Funció que comprova la mida de totes les imatges. En cas que no tinguin la mida 
#esperada(48x48), les reescala per que l'adquireixin.
def check_and_resize_images(directory, target_size=(48, 48)):
    all_correct_size = True  

    for emotion in os.listdir(directory):
        emotion_folder = os.path.join(directory, emotion)    
        if not os.path.isdir(emotion_folder):
            continue
        
        for img_file in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, img_file)         
            with Image.open(img_path) as img:
                if img.size != target_size:  #Canvia la mida en cas que no sigui la desitjada
                    all_correct_size = False
                    print(f"Resizing image: {img_path}")             
                    img = img.resize(target_size, Image.ANTIALIAS)
                    img.save(img_path)
    return all_correct_size


#Ruta als directoris corresponents per als arxius.
train_dir="/Users/pauvillen14/Documents/Q7/CPIA/PROJECTE/archive/train"
test_dir="/Users/pauvillen14/Documents/Q7/CPIA/PROJECTE/archive/test"

#Cridem la funció per comptar imatges
train_count=count_images(train_dir,'TRAIN:')
print(train_count)

test_count=count_images(test_dir,'TEST:')
print(test_count)


#Visualització d'una imatge aleatòria per a cada categoria
emotions = os.listdir(train_dir)
plt.figure(figsize=(15,10))
num = np.random.randint(1,979)

for i, emotion in enumerate(emotions, 1):
  folder = os.path.join(train_dir, emotion)
  img_path = os.path.join(folder, os.listdir(folder)[num])
  img = plt.imread(img_path)
  plt.subplot(3, 3, i)
  plt.grid()
  plt.imshow(img, cmap='gray')
  plt.title(emotion)
  plt.axis('off')
  
#Cridem la funció de la mida de les imatges i imprimim per pantalla el resultat
size_train = check_and_resize_images(train_dir, target_size=(48, 48))
if size_train:
    print("All images from 'train' are already 48x48 pixels.")
else:
    print("Some images from 'train' were resized to 48x48 pixels.")
    
size_test = check_and_resize_images(test_dir, target_size=(48, 48))
if size_train:
    print("All images from 'test' are already 48x48 pixels.")
else:
    print("Some images from 'test' were resized to 48x48 pixels.")


#%%
# Creació del model CNN

img_width, img_height = 48, 48  


#Preparem els generadors d'imatges per al model. Introduim algunes transformacions
#aleatòries a les imatges per millorar la generalització del model
train_datagen = ImageDataGenerator(  
    rescale=1.0 / 255.0,  #Normalització dels píxels
    rotation_range=30,  
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)


#Creem els generadors d'imatges anteriorment preparats. Especifiquem el directori
#d'on provenen, la mida desitjada, el batch size i la categoria corresponent.
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle = True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False
)




model = Sequential()

#Capes convolucionals per a la creació de la xarxa
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1),kernel_initializer="glorot_uniform", padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(0.01)))
model.add(Conv2D(256, (3, 3),activation='relu', padding='same', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(0.01)))
model.add(Conv2D(1024, (3, 3),activation='relu', padding='same', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Capes denses
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))



#Compilació del model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() #Retorna un resum de les capes i paràmetres del model

#Detectar si a partir d'algun moment deixa d'haver-hi millores. Es queda amb els 
#pesos més òptims d'entre totes les èpoques
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

callbacks = [earlystop] 


#Entrenament del model
history = model.fit(
    train_datagen,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=callbacks
)

#Avaluació de la precisió del model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


#%%

#Generar prediccions a partir del conjunt de test
y_true = test_generator.classes  #Categoria correcta a partir del generador d'imatges de test
y_pred_probs = model.predict(test_generator, verbose=0)  # Predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to predicted class indices

#Extreure el classification report
print("Classification report:")
target_names = list(test_generator.class_indices.keys())  
report = classification_report(y_true, y_pred, target_names=target_names)
output_file = "classification_report.txt"

#Escriure el classification report en un fitxer .txt
with open(output_file, "w") as file:
    file.write("Classification Report\n")
    file.write("=====================\n")
    file.write(report)


#Constriur la confusion matrix, per avaluar el model en funció de la categoria
print("Confusion matrix:")
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)

#Plot de la confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion matrix')
plt.show()


#%%
#Fer prediccions per comprovar el funcionament del model.


Emotion_Classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']  

#Seleccionar 20 imatges aleatòries del conjunt de test
total_images = test_generator.samples 
Random_Indices = np.random.choice(total_images, 20, replace=False) 

#Plot de les 20 imatges indicant si la seva predicció ha estat correcta o no
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    batch_index = Random_Indices[i] // batch_size  
    image_index = Random_Indices[i] % batch_size  

    
    batch_data = test_generator[batch_index]
    Random_Img = batch_data[0][image_index]  # Imagen
    Random_Img_Label = np.argmax(batch_data[1][image_index])  # Etiqueta verdadera

    #Realitzar la predicció
    Model_Prediction = np.argmax(model.predict(tf.expand_dims(Random_Img, axis=0), verbose=0))  

    ax.imshow(Random_Img.squeeze(), cmap='gray') 

    #Escriure l'emoció correcta i la predicció, en verd o vermell en funció de si ha encertat o no
    color = "green" if Emotion_Classes[Random_Img_Label] == Emotion_Classes[Model_Prediction] else "red"
    ax.set_title(f"True: {Emotion_Classes[Random_Img_Label]}\nPred: {Emotion_Classes[Model_Prediction]}", color=color)

plt.tight_layout()
plt.show()


#%%
#Guardar el model
model.save("modelo_emociones.keras")
#Mostrar en dos plots les mètriques de precissió i "loss" per al train i el test
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Test accuracy')
plt.xlabel('Epochs')
plt.xticks([0,2,4,6,8,10,12,14,16,18])
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Test loss')
plt.xlabel('Epochs')
plt.xticks([0,2,4,6,8,10,12,14,16,18])
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.show()

DOCTOR_FILE = "/Users/pauvillen14/Downloads/datos_eval_trainees_2025/datos_HCPs.csv"
ARTICLE_FILE = "/Users/pauvillen14/Downloads/datos_eval_trainees_2025/scientific_articles.xlsx"
