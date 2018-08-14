# DCGAN for Day2Night
This is a Tensorflow implementation of Alec Radford's [Unsupervised Representation Learning with Deep Convolutional Generative 
Adversarial Networks, ICLR2016](https://arxiv.org/pdf/1511.06434.pdf) for Day2Night project.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/43059677-9688883e-8e88-11e8-84a7-c8f0f6afeca6.png" width=800)
</p>
  
## Requirements
- tensorflow 1.9.0
- python 3.5.3
- numpy 1.14.2
- pillow 5.0.0
- matplotlib 2.0.2

## Generated Night Images
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43063275-5afc9842-8e96-11e8-9461-c20884c46094.png" width=800>
</p>

<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43063329-7569696c-8e96-11e8-8b10-35f65e21c8ce.png" width=800>
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
├── dcgan
│   ├── build_data.py
│   ├── dataset.py
│   ├── dcgan.py
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
**dcgan:** source codes of DCGAN  
**data:** tfrecord files for training

### Implementation Details
Implementation uses TensorFlow to train the DCGAN. Same generator and discriminator networks are used as 
described in [Alec Radford's paper](https://arxiv.org/pdf/1511.06434.pdf), except the batch normalization of training mode is 
used in training and test mode that we found to get more stalbe results and the generated image size is (128, 256).

### Training DCGAN
Use `main.py` to train a DCGAN network. Example usage:

```
python main.py --is_train=true
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
 - `save_freq`: save frequency for model, default: `10000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_size`: sample size for check generated image quality, default: `64`
 - `load_model`: folder of save model that you wish to test, (e.g. 20180704-1736). default: `None`
 
### Evaluate DCGAN
Use `main.py` to evaluate a DCGAN network. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20180704-1746
```
Please refer to the above arguments.

### Citation
```
  @misc{chengbinjin2018day2nightdcgan,
    author = {Cheng-Bin Jin},
    title = {DCGAN for Day2Night},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/GANFromAtoZ/tree/master/dcgan}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [carpedm20](https://github.com/carpedm20/DCGAN-tensorflow)
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (
email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
