import opendatasets as od
import os
import shutil
import cv2
import math
import uuid
import tensorflow as tf
from multiprocessing import Pool, cpu_count
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Register Kaggle and type in your username and Kaggle Key (Settings->API->Create New Token)
od.download("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")

# Enabling GPUs
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

# ==========================Create directory==========================

def create_folder_in_directory(directory, folder_name):
    target_folder_path = os.path.join(directory, folder_name)
    
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
        print(f"Folder '{folder_name}' created in '{directory}'.")
    else:
        print(f"Folder '{folder_name}' already exists in '{directory}'.")

directory = "./plantvillage-dataset/plantvillage dataset"
class_names = ['blight','citrus' ,'healthy', 'measles', 'mildew', 'mite', 'mold', 'rot', 'rust', 'scab', 'scorch', 'spot', 'virus']

for class_name in class_names:
    create_folder_in_directory(directory, class_name)

# ==========================Move Files==========================

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

# Check if Directory is empty or not:
def is_directory_empty(directory):
    for root, dirs, files in os.walk(directory):
        if files:  
            return False
    return True  

root = "./plantvillage-dataset/plantvillage dataset"

for origin in ['color', 'segmented', 'grayscale']:
    
    directory = os.path.join(root, origin)

    if os.path.exists(directory) == False:
        break

    for class_name in class_names:
        substring = class_name
        matching_folders = list_and_check_folders(directory, substring)
        matching_folders = matching_folders + list_and_check_folders(directory, substring.capitalize())

        # print(f"{origin}/{class_name}: ")
        # print(matching_folders)
    
        for match in matching_folders: 
            source_directory = os.path.join(directory, match)
            target_directory = os.path.join(root, class_name)
            move_all_files(source_directory, target_directory)
    if is_directory_empty(directory):
        shutil.rmtree(directory)
        print(f"Delete empty directory: {directory}")

# ==========================Augmentation==========================

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

# ==========================Load dataset==========================
def load_dataset(ds_dir,shuffle=True, batch_size=32, image_size=(128, 128), seed = 42):
    main()
    dataset = tf.keras.utils.image_dataset_from_directory(ds_dir,
                                                          shuffle=True,
                                                          batch_size=batch_size,
                                                          image_size=image_size,
                                                          seed=seed)
    return dataset

# ==========================Balance Dataset Classes==========================
def balance(args):
    with Pool(cpu_count()) as pool:
        pool.map(augment_images, args)    


def main():            
    # Initialize ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Process each class for augmentation
    root = "./plantvillage-dataset/plantvillage dataset"
    
    for class_name in class_names:
        folder_path = os.path.join(root, class_name)
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