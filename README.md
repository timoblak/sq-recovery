# sq-recovery 
The repository for project "Superquadrics and Deep learning".

## Recovery of a single rotated SQ (2020)

Implementation of paper "Learning to predict superquadric parameters from depth images with explicit and implicit supervision"

Code is located in `torch/`: 
- CNN architecture in `torch/models.py`
- Loss functions and Dataloader in `torch/classes.py`
- Training script is `torch/train.py` 
- Check `torch/visu.py` for render of SQs on GPU. Also useful for testing loss functions.  
- Test on single image: `torch/test.py`  
- Test on random set of images generated on spot: `torch/test_random.py`
- Visualize optimization with different loss functions in `torch/visu.py` 

Data related code is located in `data/`:
- Generate a dataset with `data/generation_scripts/gen_rand_rot.py`
- Example images included in `data/example_imgs`
- Pre-built binary of data renderer `data/scanner` (built for Linux) 

#### Instructions: 

Last tested with pytorch 1.9.0 on CUDA 11.1. Probably works on older versions too. 

The dataset you can generate yourself with the given scripts. 
[Download](https://unilj-my.sharepoint.com/:u:/g/personal/tim_oblak_fri1_uni-lj_si/EREnUSzH-vJKha4mv87mtIABCLhIaJ6gwbkzrFLq9w4ysg?e=cb3nfT) pretrained models for implicit and explicit supervision approaches. 
Save models to `torch/trained_models/` and run `torch/test.py`.  


## Recovery of a single SQ from isometric projection, no rotation (2019) 

This model is described in our paper "Recovery of Superquadrics from Range Images using Deep Learning: A Preliminary Study". 
It's a simple AlexNet-like architecture doing regression over 8 parameters.

Code is located in `py/`. Some instructions may be outdated. 

#### Instructions: 

Working on a Linux machine with CUDA 10 and cuDNN 7+. Python version is 3.6.x Not tested with 2.7.x  
Check requirements.txt for python packages. 

[Download](https://unilj-my.sharepoint.com/:u:/g/personal/tim_oblak_fri1_uni-lj_si/EaR_Ij6K_CtDiJIc-GS4ObwBBZRTfPu9yXRXZA2XfUCkfw?e=dmrsc7) any pretrained models. 
Run `mkdir models/` and save there.  

```
tar zxvf data_isometry_train.tar.gz -C data/
tar zxvf data_isometry_val.tar.gz -C data/
...
```

To generate data, use `data/generation_scripts/gen_rand_iso.py`
To test this use model `cnn_isometry_100k.h5` and run `python test_isometry.py`.
To train this model, run `python train_isometry.py`. Make sure, you downloaded and extracted the dataset.
 
