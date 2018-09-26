# DiscoGAN for Day2Night
This repository is a Tensorflow implementation of SKTBrain's [DiscoGAN](https://arxiv.org/abs/1703.05192), ICML2017 for Day2Night project.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/46002379-e020ed00-c0e8-11e8-81d1-3ee153c6850f.png" width=600)
</p>  
  
## Requirements
- tensorflow 1.10.0
- python 3.5.3
- numpy 1.14.2
- matplotlib 2.2.2
- pillow 5.0.0

## Applied GAN Structure
1. **Generator**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46003429-7c4bf380-c0eb-11e8-9892-c4e42eaf31e4.png" width=400>
</p>

2. **Discriminator**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46003548-c2a15280-c0eb-11e8-8b58-078f20aec279.png" width=450>
</p>

## Generated Images
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/46080278-80077500-c1d4-11e8-9c1b-d5842b8e5d7e.png" width=800>
</p>

<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/46080320-98778f80-c1d4-11e8-8202-b26d9d98de0c.png" width=800>
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
├── discoGAN
│   ├── dataset.py
│   ├── discogan.py
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
**discoGAN:** source codes of discoGAN  
**data:** tfrecord files for training

### Implementation Details
Implementation uses TensorFlow to train the DiscoGAN. Same generator and critic networks are used as described in [DiscoGAN paper](https://arxiv.org/abs/1703.05192). We applied learning rate control that started at 2e-4 for the first 1e5 iterations, and decayed linearly to zero as [cycleGAN](https://github.com/junyanz/CycleGAN). It's helpful to overcome mode collapse problem.  

To respect the original discoGAN paper we set the balance between GAN loss and reconstruction loss are 1:1. Therefore, discoGAN is not good at `A -> B -> A`. However, in the [cycleGAN](https://github.com/junyanz/CycleGAN) the ratio is 1:10. So the reconstructed image is still very similar to input image.  

The official code of [DiscoGAN](https://github.com/SKTBrain/DiscoGAN) implemented by pytorch that used weigt decay. Unfortunately, tensorflow is not support weight deacy as I know. I used regularization term instead of weight decay. So the performance maybe a little different with original one. 

### Training DiscoGAN
Use `main.py` to train a DiscoGAN network. Example usage:

```
python main.py --is_train=true
```
 - `gpu_index`: gpu index, default: `0`
 - `batch_size`: batch size for one feed forward, default: `32`
 - `dataset`: dataset name, default: `day2night`
 - `is_train`: 'training or inference mode, default: `True`
 - `learning_rate`: initial learning rate, default: `0.0002`
 - `beta1`: beta1 momentum term of Adam, default: `0.5`
 - `beta2`: beta2 momentum term of Adam, default: `0.999`
 - `weight_decay`: hyper-parameter for regularization term, default: `1e-4`
 - `iters`: number of interations, default: `100000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `10000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_batch`: number of sampling images for check generator quality, default: `200`
 - `load_model`: folder of save model that you wish to test, (e.g. 20180907-1739). default: `None` 

### Test DiscoGAN
Use `main.py` to test a DiscoGAN network. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20180926-1739
```
Please refer to the above arguments.
 
### Citation
```
  @misc{chengbinjin2018day2nightdiscogan,
    author = {Cheng-Bin Jin},
    title = {DiscoGAN for Day2Night},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/GANFromAtoZ/tree/master/discoGAN}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project refered some code from [carpedm20](https://github.com/carpedm20/DiscoGAN-pytorch) and [GunhoChoi](https://github.com/GunhoChoi/DiscoGAN-TF).  
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (
email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
