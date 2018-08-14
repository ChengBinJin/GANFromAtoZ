# pix2pix-tensorflow for Day2Night

This repository is a Tensorflow implementation of Isola's [Image-to-Image Tranaslation with Conditional Adversarial Networks, CVPR2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf). 

<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42619365-d285e190-85f2-11e8-8e52-9d53ddfc5653.png">
</p>

## Requirements
- tensorflow 1.8.0
- python 3.5.3  
- numpy 1.14.2  
- matplotlib 2.0.2  
- scipy 0.19.0

## Generated Results
**A to B**: from day image to night image  
**B to A**: from night image to day image  
<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42741388-c1db1e44-88ec-11e8-93f3-c94c2ecdb9d3.png">
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42741398-d577874e-88ec-11e8-91b4-d44dd68a22bf.png">
</p>

## Generator & Discriminator Structure
- **Generator structure**
<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42619487-2533caa6-85f3-11e8-9449-ada599622256.png" width=700>
</p>

- **Discriminator structure**
<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42619942-699a0e0c-85f4-11e8-97e0-b7403cd9abc7.png" width=400>
</p>

## Documentation
### Download Dataset
[Alderley Day/Night Dataset](https://wiki.qut.edu.au/pages/viewpage.action?pageId=181178395) is used to generate day-to-night images. Download [FRAMESA.zip](https://mega.nz/#!h1swyAwC!pWUxMnmMop8XmhaZIGjXMekVXMpi64IfI2GMADR0ako), [FRAMESB.zip](https://mega.nz/#!N9tRFLzJ!VUwj9nqpJK_L5zt-lAq3rmyP7du4RH4f1u1JIPgKA90), and [framemaches.csv](https://mega.nz/#!p1tRRYJD!rzYy1ufS_OIC4h1tJKBVEoD5P0WwcSFiTGK-q3hRPX0) files.

### Data Preparing
Day and night image alignment using supplied framematches.csv file. Then manually divide training and validation data in `train` and `val` folder.  

```
python alignment.py
```  

### Directory Hierarchy
``` 
.
├── pix2pix
│   ├── alignment.py
│   ├── dataset.py
│   ├── main.py
│   ├── pix2pix.py
│   ├── solver.py
│   ├── tensorflow_utils.py
│   └── utils.py
├── data
│   ├── paired
│   │   ├── train
│   │   └── val
├── ...
```  
**pix2pix:** source codes of pix2pix
**data/paired:** paired data folder for training and validation data

### Training pix2pix
Use `main.py` to train a vanilla GAN network. Example usage:

```
python main.py --is_train=true --which_direction=0
```
 - `gpu_index`: gpu index, default: `0`
 - `dataset`: dataset name, default: `paired`
 - `which_direction` AtoB (0) or BtoA (1), default: `0`
 - `batch_size`: batch size for one feed forward, default: `1`
 - `is_train`: 'training or inference mode, default: `False`
 - `learning_rate`: initial learning rate, default: `0.0002`
 - `beta1`: momentum term of Adam, default: `0.5`
 - `iters`: number of interations, default: `200000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `10000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_size`: sample size for check generated image quality, default: `4`
 - `load_model`: folder of save model that you wish to test, (e.g. 20180704-1736). default: `None`
 
### Evaluate pix2pix
Use `main.py` to evaluate a vanilla GAN network. Example usage:

```
python main.py --is_train=false --which_direction=0/or/1? --load_model=folder/you/wish/to/test/e.g./20180704-1746
```
Please refer to the above arguments.

### Citation
```
  @misc{chengbinjin2018day2nightpix2pix,
    author = {Cheng-Bin Jin},
    title = {pix2pix for Day2Night},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/GANFromAtoZ/tree/master/pix2pix}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [yenchenlin](https://github.com/yenchenlin/pix2pix-tensorflow) and [pix2pix official websit](https://phillipi.github.io/pix2pix/)
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

### License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
