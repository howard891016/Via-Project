# Via-Project
Code used in project

## python/segment_model_augment.py
Import this python file into your notebook to load training dataset.
- How to use
  - Put this file at the same level of your notebook.
  - Register an account on kaggle and create a token to download the dataset. (Settings->API->Create New Token)
  - Then just run the code below and enter your username and token to download the dataset and train your model.
    ```
    python3 segment_model_augment.py
    ```
  - If you want to train segmented image, just change the train_dir and origin variable from 'color' to 'segment' in the main function.
  - Name your model by just changing the input of the save_model function used in the main function.


## .keras 
Just load into your notebook and you can use it to predict images.
```
model = tf.keras.models.load_model('model name')
```
Use 'model.summary()' to check the model's input size.

## .tflite
Tflite models can be used to convert to mdla on transforma using the command below.
```
ncc-tflite --arch mdla3.0 --relax-fp32 --opt-accuracy -O 3 model.tflite -o model.mdla
```

## best_models
Finetuned Detection models that uses dataset shot by esp32 camera.
