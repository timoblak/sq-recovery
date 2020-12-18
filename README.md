# sq-recovery 
The repository for project "Superquadrics and Deep learning".

## Recovery of a single rotated SQ (2020)

Implementation of paper "Learning to predict superquadric parameters from depth images with explicit and implicit supervision"

Code is located in `torch/`: 
- CNN architecture in `torch/models.py`
- Loss functions and Dataloader in `torch/classes.py`
- Training script is `torch/train.py` 
- Generate a dataset with `data/generation_scripts/gen_rand_rot.py`
- Check `torch/visu.py` for render of SQs on GPU. Also useful for testing loss functions.  

#### Instructions: 

Last tested with pytorch 1.3.1 on CUDA 10. Probably works on newer versions too. 

Should you need the original dataset or trained models, email me at tim.oblak [at] fri.uni-lj.si.

## Recovery of a single SQ from isometric projection, no rotation (2019) 

This model is described in our paper "Recovery of Superquadrics from Range Images using Deep Learning: A Preliminary Study". 
It's a simple AlexNet-like architecture doing regression over 8 parameters.

Code is located in `py/`. Some instructions may be outdated. 

#### Instructions: 

Working on a Linux machine with CUDA 10 and cuDNN 7+. Python version is 3.6.x Not tested with 2.7.x  
Check requirements.txt for python packages. 

[Download](https://unilj-my.sharepoint.com/:f:/g/personal/to1702_student_uni-lj_si/EkHPlRx2AatEvDwARAhYsqkBXBkkdXWji1qNYcN-nwrZZw?e=bTYkZT) any pretrained models. 
Run `mkdir models/` and save there. 


[Download](https://unilj-my.sharepoint.com/:f:/g/personal/to1702_student_uni-lj_si/EjNoQpybF-xIuL-9dDlur14B1NLWrK1XWdAcHnmvsS7ecg?e=WpmY0J) and extract all data folders. 

```
tar zxvf data_isometry_train.tar.gz -C data/
tar zxvf data_isometry_val.tar.gz -C data/
...
```

To test this use model `cnn_isometry_100k.h5` and run `python test_isometry.py`.
To train this model, run `python train_isometry.py`. Make sure, you downloaded and extracted the dataset.
 
