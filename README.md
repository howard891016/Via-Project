# Via-Project
Code used in project

## kaggle_dataset.py
Import this python file into your notebook to load training dataset.
- How to use
  - Put this file at the same level of your notebook.
  - Register an account on kaggle and create a token to download the dataset. (Settings->API->Create New Token)
  - Use below code to load dataset. Decide the parameter yourself or you can just call load_dataset() and it will return a default dataset to you.
    ```
    from kaggle_dataset import *
    dataset = load_dataset(shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, seed=42)
    ```

## new_detect_128_v3.keras
Just load into your notebook and you can use it to predict images.
```
model = tf.keras.models.load_model('new_detect_128_v3.keras')
```
Use 'model.summary()' to check the model's input size.
