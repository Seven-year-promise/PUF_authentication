# PUF_authentication

## Introduction:
This is the implementation for our paper 'Random fractal-enabled physical unclonable functions with dynamic AI authentication'

## Dataset:
The dataset is located in the folder "./data", where 
"all_rgb": original images of the PUF patterns
"all": grayscale images, 
"all_render_bg": images after the preprocessing 
"annotations": json files for training, adding, and testing the method
"features": the CNN features generated for the base dataset
"similarity": images used for the computation of similarity

# Running the code
## Create the environment using the file "puf_authentication.yml"

## Training the initial base model
"python train_class_only_init.py"

## Update the model with newly added PUF patterns
"python train_class_only_add.py --a_n_classes 200 --acc_thre 95 --batch_size 2000"

## Generate the test dataset before testing
"python generate_test_set.py"

## Test the model
"python test_all_pipeline.py"

For cite of thsi work:
@article{sun2022random,
  title={Random fractal-enabled physical unclonable functions with dynamic AI authentication},
  author={Sun, Ningfei and Chen, Ziyu and Wang, Yanke and Wang, Shu and Xie, Yong and Liu, Qian},
  year={2022}
}


 

	