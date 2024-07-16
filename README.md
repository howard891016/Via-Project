# Via-Project
Code used in project

## kaggle_dataset.py
Import this python file into your notebook to load training dataset.
- How to use
  - Put this file at the same level of your notebook.
  - Register an account on kaggle and create a token to download the dataset. (Settings->API->Create New Token)
  - Use below code to load dataset. The train_dir is where you 
    ```
    from kaggle_dataset import *
    dataset = load_dataset(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE, seed=42)
    ```
