# WGAN-GP for Day2Night
This repository is a Tensorflow implementation of Ishaan Gulrajani's [WGAN-GP](https://arxiv.org/abs/1704.00028) for Day2Night project.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/46613989-a53ea080-cb4f-11e8-83c1-a8b99dc7bc5b.png" width=800>
</p>

## Requirements
- tensorflow 1.10.0
- python 3.5.5
- numpy 1.14.2
- pillow 5.0.0
- matplotlib 2.2.2

## Generated Night Images
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/49413357-03d07580-f7b3-11e8-9ffa-67ac9ba21f0c.png" width=900>
</p>
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/49413386-1c409000-f7b3-11e8-9942-b0c4143fae4b.png" width=900>
</p>

## Documentation
### Download Dataset
[Alderley Day/Night Dataset](https://wiki.qut.edu.au/pages/viewpage.action?pageId=181178395) is used to generate day-to-night images. 
Download [FRAMESA.zip](https://mega.nz/#!h1swyAwC!pWUxMnmMop8XmhaZIGjXMekVXMpi64IfI2GMADR0ako), 
[FRAMESB.zip](https://mega.nz/#!N9tRFLzJ!VUwj9nqpJK_L5zt-lAq3rmyP7du4RH4f1u1JIPgKA90), and 
[framemaches.csv](https://mega.nz/#!p1tRRYJD!rzYy1ufS_OIC4h1tJKBVEoD5P0WwcSFiTGK-q3hRPX0) files.

### Data Preparing
Encode original data to tfrecrod files.  
```
python build_data.py --input_dataA=YOUR_DATA_A_DIR \
  --input_dataB=YOUR_DATA_B_DIR \
  --output_dataA=alderley_day \
  --output_dataB=alderley_night \
  --extension=.jpg
```  
tfrecord files are writed in ../data/tfrecords folder as shown in Directory Hierarchy.
Check ```python build_data.py --help``` for more information.

- `--input_dataA`: data A input directory, default: `None`
- `--input_dataB`: data B input directory, default: `None`
- `--output_dataA`: data A output directory, default: `None`
- `--output_dataB`: data B output directory, default: `None`
- `--extension`: input image extension, default: `.png`

### Directory Hierarchy
``` 
.
├── wgan-gp
│   ├── build_data.py
│   ├── dataset.py
│   ├── main.py
│   ├── reader.py
│   ├── solver.py
│   ├── tensorflow_utils.py
│   ├── utils.py
│   └── wgan_gp.py
├── data
│   ├── tfrecords
│   │   ├── alderley_day.tfrecords
│   │   └── alderley_night.tfrecords
├── ...
```  
**wgan-gp:** source codes of WGAN-GP  
**data:** tfrecord files for training

### Training WGAN-GP
Use `main.py` to train a WGAN-GP network. Example usage:

```
python main.py
```
 - `gpu_index`: gpu index, default: `0`  
 - `batch_size`: batch size for one feed forward, default: `8`  
 - `dataset`: dataset name, default: `day2night`  
 
 - `is_train`: training or inference mode, default: `True`  
 - `learning_rate`: initial learning rate for Adam, default: `0.0001`  
 - `num_critic`: the number of iterations of the critic per generator iteration, default: `5`
 - `z_dim`: dimension of z vector, default: `128`
 - `lambda_`: gradient penalty lambda hyperparameter, default: `10.`
 - `beta1`: beta1 momentum term of Adam, default: `0.5`
 - `beta2`: beta2 momentum term of Adam, default: `0.9`

 - `iters`: number of interations, default: `200000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `10000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_batch`: number of sampling images for check generator quality, default: `8`
 - `load_model`: folder of save model that you wish to test, (e.g. 20181120-1558). default: `None` 
 
### WGAN-GP During Training
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/49414133-9ffb7c00-f7b5-11e8-835e-b4412dde0b03.png" width=900>
</p>

### Evaluate WGAN-GP
Use `main.py` to test a WGAN-GP network. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20181121-2049
```
Please refer to the above arguments.

### Citation
```
  @misc{chengbinjin2018day2nightwgan-gp,
    author = {Cheng-Bin Jin},
    title = {WGAN-GP for Day2Night},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/GANFromAtoZ/tree/master/wgan-gp}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [igul222](https://github.com/igul222/improved_wgan_training).
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer).

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
