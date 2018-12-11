# DPNN-Food_Classification

### Installation
if you want to use pretrained models, then all you need to do is:
```sh
git clone https://github.com/djang000/Food-Classication.git
```

if you also want to train new modes, you will need the Food-101 or other natural images for training files and MobileNet wegihts by running.

you can download Mobienet_V2 weight from below website
```sh
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
```
To prepare the Food dataset for use with train.py, you will have to convert it to Tensorflow's TFRecords format, which shards the images into large files for more efficient reading from disk. convert_tfrecord.py can be used for this as shown below. Change --num_threads to however many threads your cores can handle, and ensure that it divides whatever you choose for --train_shards. This block will give shards ~100MB in size:
But before the below command, you should change surfix in config.py train or test and TFrecord_dir for saving conveted tfrecord files. 

```sh
python convert_tfrecord.py  
```

### Usage

Following are examples of how the scripts in this repo can be used.

- eval.ipynb

	you can show the evaluation result using trained model.

- train.py

	you should run below commad after change config parameters in config.py

	Tensorboard logs of the loss functions and checkpoints of the model are also created. 		Note that this will take a long time to get a good result. Example usage:

```sh
python train.py
```

