import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import math
import shutil
from pathlib import Path
import uuid
import opendatasets as od
from multiprocessing import Pool, cpu_count
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================Functions============================

def Enable_GPU():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

def create_folder_in_directory(directory, folder_name):
    target_folder_path = os.path.join(directory, folder_name)
    
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
        print(f"Folder '{folder_name}' created in '{directory}'.")
    else:
        print(f"Folder '{folder_name}' already exists in '{directory}'.")

def download_dataset(directory, class_names):
    od.download("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")

    for class_name in class_names:
        create_folder_in_directory(directory, class_name)

# File Categorize

# Move file function:
def move_all_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    files = os.listdir(source_dir)
    
    for file_name in files:
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        
        if os.path.isfile(source_file):
            shutil.move(source_file, target_file)

# Mapping disease name function:
def list_and_check_folders(directory, substring):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    matching_folders = [folder for folder in folders if substring in folder]
    return matching_folders

def is_directory_empty(directory_path):
    return not any(os.listdir(directory_path))

# File Migration

def file_migration(root, origin, class_names):
    
    directory = os.path.join(root, origin)
    
    if os.path.exists(directory) == False:
        print(f"{directory} doesn't exist")
    
    for class_name in class_names:
        substring = class_name
        matching_folders = list_and_check_folders(directory, substring)
        matching_folders = matching_folders + list_and_check_folders(directory, substring.capitalize())
    
        for match in matching_folders: 
            source_directory = os.path.join(directory, match)
            target_directory = os.path.join(directory, class_name)
            move_all_files(source_directory, target_directory)
    
    for file in os.listdir(directory):
        dir = os.path.join(directory, file)
        if is_directory_empty(dir):
            shutil.rmtree(dir)
            print(f"Delete empty directory: {dir}")

def augment_images(args):
    class_name, folder_path, file_name, times, datagen = args
    file_path = os.path.join(folder_path, file_name)
    img = cv2.imread(file_path)
    if img is None:
        print(f"Failed to load image: {file_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((1,) + img.shape)  # Batch
    
    # Generate augmented images
    for _ in range(times):
        for batch in datagen.flow(img, batch_size=1, save_prefix='aug', save_format='jpg'):
            # Generate a unique filename
            unique_filename = f"aug_{uuid.uuid4()}.jpg"
            # save_path = './new_augment/' + class_name + '/' + unique_filename
            save_path = os.path.join(folder_path, unique_filename)
            cv2.imwrite(save_path, batch[0])
            break 

def balance(args):
    with Pool(cpu_count()) as pool:
        pool.map(augment_images, args)    

# ============================Data Balance============================
def data_balance(train_dir, class_names, datagen):
    for class_name in class_names:
        folder_path = os.path.join(train_dir, class_name)
        file_names = os.listdir(folder_path)
        file_num = len(file_names)
        
        times = 0
        if file_num < 10000:
            times = math.floor(10000 / file_num)
        if class_name == 'healthy':
            times = 1
        
        if times == 0:
            continue
        
        args = [(class_name, folder_path, file_name, times, datagen) for file_name in file_names]
        
        print(f"Start {class_name} Augmentation")
        # Use multiprocessing to parallelize the augmentation
        balance(args)
        print(f"Finish {class_name} Augmentation, file num: {len(os.listdir(folder_path))}")

# ============================Dataset Process============================
def dataset_process_and_load(train_dir, batch_size, image_size, seed, validation_split):
    # Shuffled
    dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=batch_size,
                                                            image_size=image_size,
                                                            seed=seed)
    
    validation_split = validation_split # 0.2
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    print(dataset.class_names)
    class_names = dataset.class_names
    
    # # 分割數據集
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)
    
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)
    
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset, train_dataset, validation_dataset, test_dataset

# ============================Build Model============================
def build_model(IMG_SIZE, dataset):
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),
    ])
    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    image_batch, label_batch = next(iter(dataset))
    feature_batch = base_model(image_batch)
    
    base_model.trainable = False
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    
    fix_layer = tf.keras.layers.Dense(256, activation='relu')
    fix_batch = fix_layer(feature_batch_average)
    
    want_layer = tf.keras.layers.Dense(128, activation='relu')
    want_batch = want_layer(fix_batch)
    
    prediction_layer = tf.keras.layers.Dense(13, activation='softmax')
    prediction_batch = prediction_layer(want_batch)
    
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = fix_layer(x)
    x = want_layer(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    model.summary()
    
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model, base_model

def train_model(model, train_dataset, validation_dataset, test_dataset, initial_epochs):
    initial_epochs = initial_epochs
    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)
    return model

# ============================Fine Tune============================
def fine_tune(model, base_model, train_dataset, validation_dataset, test_dataset, initial_epochs, fine_tune_epochs, fine_tune_at):
    base_model.trainable = True
    
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))
    
    # Fine-tune from this layer onwards
    fine_tune_at = fine_tune_at
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable = False
    
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / 10),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]  
    )
    
    model.summary()
    
    fine_tune_epochs = fine_tune_epochs
    total_epochs =  initial_epochs + fine_tune_epochs
    
    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=len(history.epoch),
                             validation_data=validation_dataset)
    
    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)
    return model

# ============================Save Model============================
def save_model(model, model_name):
    tf.keras.models.save_model(model, model_name)

# ============================Main============================

def main():
    # ============================Path============================

    root = "./plantvillage-dataset/plantvillage dataset"
    origin = 'segmented'
    train_dir = './plantvillage-dataset/plantvillage dataset/segmented'
    class_names = ['blight','citrus' ,'healthy', 'measles', 'mildew', 'mite', 'mold', 'rot', 'rust', 'scab', 'scorch', 'spot', 'virus']
    
    datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
    )
    
    BATCH_SIZE = 32
    IMG_SIZE = (128, 128)
    initial_epochs = 10
    fine_tune_epochs = 10
    seed = 42
    validation_split = 0.2
    fine_tune_at = 100
    
    Enable_GPU()
    download_dataset(train_dir, class_names)
    file_migration(root, origin, class_names)
    data_balance(train_dir, class_names, datagen)
    dataset, train_dataset, validation_dataset, test_dataset = dataset_process_and_load(train_dir, BATCH_SIZE, IMG_SIZE, seed, validation_split)
    model, base_model = build_model(IMG_SIZE, dataset)
    model = train_model(model, train_dataset, validation_dataset, test_dataset, initial_epochs)
    model = fine_tune(model, base_model, train_dataset, validation_dataset, test_dataset, initial_epochs, fine_tune_epochs, fine_tune_at)
    save_model(model, "detection.keras")