# CycleGAN TensorFlow
This work is a TensorFlow implementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf). This project got a lot of inspiration from [vanhuyz's CycleGAN-TensorFlow](https://github.com/vanhuyz/CycleGAN-TensorFlow) work. 

![picture3](https://user-images.githubusercontent.com/37034031/39303992-1b99404e-4993-11e8-8bd5-8ae4dc557847.png)

## Package Dependency
- tensorflow 1.17.0
- python 3.5.3
- numpy 1.14.2
- matplotlib 2.0.2
- pillow 5.0.0
- opencv 3.2.0

## Download Dataset
[Alderley Day/Night Dataset](https://wiki.qut.edu.au/pages/viewpage.action?pageId=181178395) is used to generate day-to-night images. Download [FRAMESA.zip](https://mega.nz/#!h1swyAwC!pWUxMnmMop8XmhaZIGjXMekVXMpi64IfI2GMADR0ako), [FRAMESB.zip](https://mega.nz/#!N9tRFLzJ!VUwj9nqpJK_L5zt-lAq3rmyP7du4RH4f1u1JIPgKA90), and [framemaches.csv](https://mega.nz/#!p1tRRYJD!rzYy1ufS_OIC4h1tJKBVEoD5P0WwcSFiTGK-q3hRPX0) files.

## Data Preparing
Encode original data to tfrecrod files.  
```
python build_data.py --input_dataA=YOUR_DATA_A_DIR --input_dataB=YOUR_DATA_B_DIR --output_dataA=alderley_day --output_dataB=alderley_night --extension=.jpg
```  
tfrecord files are writed in ../data/tfrecords folder as shown in Directory Hierarchy.
Check ```python build_data.py --help``` for more information.  

## Directory Hierarchy
``` 
.
├── cycleGAN
│   ├── build_data.py
│   ├── cycle_gan.py
│   ├── dataset.py
│   ├── main.py
│   ├── reader.py
│   ├── solver.py
│   ├── TensorFlow_utils.py
│   ├── utils.py
│   └── video2frames.py
├── data
│   ├── tfrecords
│   │   ├── alderley_day.tfrecords
│   │   └── alderley_night.tfrecords
```  
**cycleGAN:** source codes of cycleGAN  
**data:** tfrecord files for training

## Training
Move to **cycleGAN** folder and run main.py
```
python main.py 
```
### Arguments
**gpu_index:** gpu index if you have multiple gpus, default: 0  
**batch_size:** batch size, default: 1  
**dataset:** dataset name, default: day2night  
**is_train:** training or inference mode, default: True (training mode)  

**learning_rate:** initial learning rate for Adam, default: 2e-4  
**beta1:** momentum term of Adam, default: 0.5  
**iters:** number of iterations, default: 2e+5  
**print_freq:** print frequency for loss, default: 100  
**save_freq:**  save frequency for model, default: 1000  
**sample_freq:** sample frequency for saving image, default: 200  
**load_model:**  folder of saved model that you wish to continue training (e.g. 20180502-1610), default: None  


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/vanhuyz/CycleGAN-TensorFlow/blob/master/LICENSE) for more details.

## References
- CycleGAN paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)  
- [Vanhuyz's CycleGAN TensorFlow implementation](https://github.com/vanhuyz/CycleGAN-TensorFlow)
