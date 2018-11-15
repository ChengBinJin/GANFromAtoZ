# VAE for Day2Night
This repository is a Tensorflow implementation of Variationl Auto-Enocer ([VAE, 
Auto-Encoding Variational Bayes, ICLR2014](https://arxiv.org/pdf/1312.6114.pdf)) by Kingma et al. for Day2Night project.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/48118604-176de680-e2b0-11e8-9e9e-c4be981465f1.png" width=700)
</p>  

## Requirements
- tensorflow 1.10.0
- python 3.5.5
- numpy 1.14.2
- scipy 0.19.1
- matplotlib 2.2.2

## Applied VAE Structure
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/48537360-b6b85c80-e8f4-11e8-9e9a-2950a1e74476.png" width=800>
</p>

## Generated Images
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/48538140-fd0ebb00-e8f6-11e8-93b4-ce56a569c28f.png" width=600>
</p>

<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/48538157-0b5cd700-e8f7-11e8-9ad7-f775a35962fe.png" width=600>
</p>  

**Note:** Implemented VAE applied fully connection layers that caused the quality of the generatd image is bad. 
It seems that the current model failed to capture the latent structure of the night images.

## Documentation
### Download Dataset
[Alderley Day/Night Dataset](https://wiki.qut.edu.au/pages/viewpage.action?pageId=181178395) is used to generate day-to-night images. 
Download [FRAMESA.zip](https://mega.nz/#!h1swyAwC!pWUxMnmMop8XmhaZIGjXMekVXMpi64IfI2GMADR0ako), 
[FRAMESB.zip](https://mega.nz/#!N9tRFLzJ!VUwj9nqpJK_L5zt-lAq3rmyP7du4RH4f1u1JIPgKA90), 
and [framemaches.csv](https://mega.nz/#!p1tRRYJD!rzYy1ufS_OIC4h1tJKBVEoD5P0WwcSFiTGK-q3hRPX0) files.

### Data Preparing
Day and night image alignment using supplied framematches.csv file. Then manually divide training and validation data in `train` 
and `val` folder.  

```
python alignment.py
```  

### Directory Hierarchy
``` 
.
├── VAE
│   ├── alignment.py
│   ├── dataset.py
│   ├── main.py
│   ├── solver.py
│   ├── tensorflow_utils.py
│   ├── utils.py
│   └── vae.py
├── data
│   ├── paired
│   │   ├── train
│   │   └── val
├── ...
```  
**VAE:** source codes of VAE  
**data/paired:** paired data folder for training and validation data  

### Training VAE
Use `main.py` to train a VAE network. Example usage:

```
python main.py
```
 - `gpu_index`: gpu index, default: `0`
 - `batch_size`: batch size for one feed forward, default: `32`
 - `dataset`: dataset name default: `paired`
 
 - `is_train`: training or inference mode, default: `True`
 - `learning_rate`: initial learning rate for Adam, default: `0.001`
- `z_dim`: dimension of z vector, default: `128`

 - `iters`: number of interations, default: `20000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `5000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_batch`: number of sampling images for check generator quality, default: `8`
 - `load_model`: folder of save model that you wish to test, (e.g. 20181108-1029). default: `None` 
 
### Test VAE
Use `main.py` to test a VAE network. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20181112-0804
```
Please refer to the above arguments.  

### Citation
```
  @misc{chengbinjin2018day2nightvae,
    author = {Cheng-Bin Jin},
    title = {vae for Day2Night},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/GANFromAtoZ/tree/master/VAE}},
    note = {commit xxxxxxx}
  }
```
### Attributions/Thanks
- This project refered some code from [hwalsuklee](https://github.com/hwalsuklee/tensorflow-mnist-VAE).  
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

### License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: 
sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
