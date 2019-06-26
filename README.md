# SNIP
This repository contains code for the paper [SNIP: Single-shot Network Pruning based on Connection Sensitivity (ICLR 2019)](https://arxiv.org/abs/1810.02340).

## Prerequisites

### Dependencies
* tensorflow < 2
* python 2.7 or python 3.6
* packages in `requirements.txt`

### Datasets
Put the following datasets in your preferred location (e.g., `./data`).
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Usage
To run the code (LeNet on MNIST by default):
```
python main.py --path_data=./data
```
For example, in order to reproduce results for VGG-D:
```
python main.py --logdir ./reproduce-vgg --path_data ./data --datasource cifar-10 --aug_kinds fliplr translate_px --arch vgg-d --target_sparsity 0.95 --batch_size 128 --train_iterations 150000 --optimizer momentum --lr_decay_type piecewise --decay_boundaries 30000 60000 90000 120000 --decay_values 0.1 0.02 0.004 0.0008 0.00016
```
See `main.py` to run with other options.

## Citation
If you use this code for your work, please cite the following:
```
@inproceedings{lee2018snip,
  title={SNIP: Single-shot network pruning based on connection sensitivity},
  author={Lee, Namhoon and Ajanthan, Thalaiyasingam and Torr, Philip HS},
  booktitle={ICLR},
  year={2019},
}
```

## License
This project is licensed under the MIT License.
See the [LICENSE](https://github.com/namhoonlee/snip-public/blob/master/LICENSE) file for details.
