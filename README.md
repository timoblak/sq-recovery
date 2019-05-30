## Superquadric recovery 
The repository for project "Superquadrics and Deep learning".

#### Description


#### Instructions: 

- Prepare the environment 

Working on a machine with CUDA 9 and cuDNN 7.
Python version is 3.5. Not tested with 2.7.  

Check the requirements.txt for python packages. 


- Uncompress data folders 

```
tar zxvf data/compressed_data/data_isometry_train.tar.gz -C data/
tar zxvf data/compressed_data/data_isometry_val.tar.gz -C data/
```

- To train the isometric model
```
cd py
python train.py 
```

- To test the isometric model
```
cd py
python test_isometry.py 
```