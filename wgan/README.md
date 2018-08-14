# WGAN for Day2Night
This repository is a Tensorflow implementation of Martin Arjovsky's 
[Wasserstein GAN, arXiv:1701.07875v3](https://arxiv.org/pdf/1701.07875.pdf) for Day2Night project.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/43870865-b795a83e-9bb4-11e8-8005-461951b3d7b7.png" width=700)
</p>

## Requirements
- tensorflow 1.9.0
- python 3.5.3
- numpy 1.14.2
- pillow 5.0.0
- scipy 0.19.0
- matplotlib 2.2.2

## Applied GAN Structure
1. **Generator (DCGAN)**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/43059677-9688883e-8e88-11e8-84a7-c8f0f6afeca6.png" width=700>
</p>

2. **Critic (DCGAN)**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/43060075-47f274d0-8e8a-11e8-88ff-3211385c7544.png" width=500>
</p>

## Generated Night Images
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/44075067-44887e9a-9fd6-11e8-821d-d11514c7213f.png" width=800>
</p>
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/44075109-65d1009a-9fd6-11e8-8f2d-f5ec5e4e4919.png" width=800>
</p>

**Note:** The results are not good as paper mentioned. 
We found that the Wasserstein distance can't converge well in the Day2Night dataset.
High dimension of the data maybe the problem.

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
├── wgan
│   ├── build_data.py
│   ├── dataset.py
│   ├── wgan.py
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
**wgan:** source codes of WGAN  
**data:** tfrecord files for training

### Implementation Details
Implementation uses TensorFlow to train the WGAN. 
Same generator and critic networks are used as described in [Alec Radford's paper](https://arxiv.org/pdf/1511.06434.pdf). 
WGAN does not use a sigmoid function in the last layer of the critic, a log-likelihood in the cost function. 
Optimizer is used RMSProp instead of Adam.  

### Training WGAN
Use `main.py` to train a WGAN network. Example usage:

```
python main.py --is_train=true
```
 - `gpu_index`: gpu index, default: `0`
 - `batch_size`: batch size for one feed forward, default: `64`
 - `dataset`: dataset name, default: `day2night`
 - `is_train`: 'training or inference mode, default: `False`
 - `learning_rate`: initial learning rate, default: `0.00005`
 - `num_critic`: the number of iterations of the critic per generator iteration, default: `5`
 - `clip_val`: clipping value for Lipschitz costrain of the WGAN, default: `0.01`
 - `z_dim`: dimension of z vector, default: `100`
 - `iters`: number of interations, default: `100000`
 - `print_freq`: print frequency for loss, default: `50`
 - `save_freq`: save frequency for model, default: `10000`
 - `sample_freq`: sample frequency for saving image, default: `250`
 - `sample_size`: sample size for check generated image quality, default: `16`
 - `load_model`: folder of save model that you wish to test, (e.g. 20180704-1736). default: `None`
 
 ### Wasserstein Distance During Training
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/44075809-0b40c892-9fd9-11e8-8cdb-671a99c80a4f.png" width=900>
</p>

 ### Evaluate WGAN
Use `main.py` to evaluate a WGAN network. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20180704-1746
```
Please refer to the above arguments.
 
### Citation
```
  @misc{chengbinjin2018day2nightwgan,
    author = {Cheng-Bin Jin},
    title = {WGAN for Day2Night},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/GANFromAtoZ/tree/master/wgan}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [wiseodd](https://github.com/wiseodd/generative-models)
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (
email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
 
 
