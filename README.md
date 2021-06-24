**2d segmentation for LITS2017** (the challenge website in [codalab](https://competitions.codalab.org/competitions/17094))
--


# How to use it 
step 1 data process
--   
``python data_prepare/preprocess.py``

you can get the data like this,each npy file shape is ``448*448*3``,use each slice below and above as input image,the mid slice as mask(``448*448``)

```
data---
    trainImage_k1_1217---
        1_0.npy
        1_1.npy
        ......
    trainMask_k1_1217---
        1_0.npy
        1_1.npy
        ......
```

step 2 Train the model 
--
``python train.py``

step 3 Test the model 
--

``python test.py``


# Result(Test)
| Method     |U-Net  |
| :----------:|:----:|
| `Dice(liver)`|0.951|
| `Dice(tumor)`|0.613|

