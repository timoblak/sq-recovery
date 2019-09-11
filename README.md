## Superquadric recovery 
The repository for project "Superquadrics and Deep learning".


#### Instructions: 

- Prepare the environment 

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

- Isometric model

This is the version of the model, described in our latest paper "Recovery of Superquadrics from Range Images
using Deep Learning: A Preliminary Study". It's a simple AlexNet-like architecture doing regression over 8 parameters. 

To test this use model `cnn_isometry_100k.h5` and run `python test_isometry.py`.

To train this model, run `python train_isometry.py`. Make sure, you downloaded and extracted the dataset.

- Randomized rotations with quaternions 



Currently in development. Working on a differentiable renderer/loss function.  
  Experimental scripts (renderer, batched renderer, tf implementation) in `experiments/`.
  Final implementation in `loss_functions.py`. To train, run `python train_rotation.py`.

TODO: clipping of predicted parameters, quaternion normalization, Lagrange multipliers 
