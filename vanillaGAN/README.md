# Vanilla GAN for Day2Night

This is Tensorflow implementation of the [Vanilla GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) 
(Ian J. Goodfellows, et al., "Generative Adversarial Nets," NIPS2014) for day2night project.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/42316086-e7223f64-8083-11e8-8653-2e9e52bf3e79.png" width=600)
</p>
  
## Requirements
- tensorflow 1.18.0
- python 3.5.3
- numpy 1.14.2
- pillow 5.0.0

## Applied GAN Structure
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/42319346-a534df18-808c-11e8-8f7b-018c6a9b51dd.png" width=800>
</p>

## Generated Night Images
The following generated results are very bad. One reason is that we applied a shallow network, 2 fully layer network, and another reason maybe the big image dimension. The generated image size is 128x256x3.
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/42316868-d8d6e44e-8085-11e8-91c8-d02e5af74308.png" width=800>
</p>

## Documentation
### Download Dataset
[Alderley Day/Night Dataset](https://wiki.qut.edu.au/pages/viewpage.action?pageId=181178395) is used to generate day-to-night images. Download [FRAMESA.zip](https://mega.nz/#!h1swyAwC!pWUxMnmMop8XmhaZIGjXMekVXMpi64IfI2GMADR0ako), [FRAMESB.zip](https://mega.nz/#!N9tRFLzJ!VUwj9nqpJK_L5zt-lAq3rmyP7du4RH4f1u1JIPgKA90), and [framemaches.csv](https://mega.nz/#!p1tRRYJD!rzYy1ufS_OIC4h1tJKBVEoD5P0WwcSFiTGK-q3hRPX0) files.

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
├── vanillaGAN
│   ├── build_data.py
│   ├── vanillaGAN.py
│   ├── dataset.py
│   ├── main.py
│   ├── reader.py
│   ├── solver.py
│   ├── tensorflow_utils.py
│   └── utils.py
├── data
│   ├── tfrecords
│   │   ├── alderley_day.tfrecords
│   │   └── alderley_night.tfrecords
├── ...
```  
**vanillaGAN:** source codes of vanillaGAN
**data:** tfrecord files for training

### Training Vanilla GAN
Use `main.py` to train a vanilla GAN network. Example usage:

```
python main.py --is_train true
```
 - `gpu_index`: gpu index, default: `0`
 - `batch_size`: batch size for one feed forward, default: `64`
 - `dataset`: dataset name, default: `day2night`
 - `is_train`: 'training or inference mode, default: `False`
 - `learning_rate`: initial learning rate, default: `0.0002`
 - `beta1`: momentum term of Adam, default: `0.5`
 - `z_dim`: dimension of z vector, default: `100`
 - `iters`: number of interations, default: `200000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `1000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_size`: sample size for check generated image quality, default: `16`
 - `load_model`: folder of save model that you wish to test, (e.g. 20180704-1736). default: `None`

### Evaluate Vanilla GAN
Use `main.py` to evaluate a vanilla GAN network. Example usage:

```
python main.py --is_train false --load_model folder/you/wish/to/test/e.g./20180704-1746
```
Please refer to the above arguments.

### Citation
```
  @misc{chengbinjin2018day2nightvanillagan,
    author = {Cheng-Bin Jin},
    title = {Vanilla GAN for Day2Night},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/GANFromAtoZ/tree/master/vanillaGAN/}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [Vanhuyz](https://github.com/vanhuyz/CycleGAN-TensorFlow)
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

### License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
