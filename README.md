# MGVTON
**Unofficial PyTorch reproduction of MGVTON.**


I'm reproducing [MGVTON] (https://arxiv.org/pdf/1902.11026.pdf). The implementation is based on the reproduction of SWAPNET (https://github.com/andrewjong/SwapNet)

## Contributing

The following instruction is from author of SwapNet:

# Installation

This repository is built with PyTorch. I recommend installing dependencies via [conda](https://docs.conda.io/en/latest/).

With conda installed run:
```
cd SwapNet/
conda env create  # creates the conda environment from provided environment.yml
conda activate swapnet
```
Make sure this environment stays activated while you install the ROI library below!

## Install ROI library (required)
I borrow the ROI (region of interest) library from [jwyang](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0). This must be installed for this project to run. Essentially we must 1) compile the library, and 2) create a symlink so our project can find the compiled files.

**1) Build the ROI library**
```
cd ..  # move out of the SwapNet project
git clone https://github.com/jwyang/faster-rcnn.pytorch.git # clone to a SEPARATE project directory
cd faster-rcnn.pytorch
git checkout pytorch-1.0
pip install -r requirements.txt
cd lib/pycocotools
```
Important: now COMPLETE THE INSTRUCTIONS [HERE](https://github.com/jwyang/faster-rcnn.pytorch/issues/402#issuecomment-448485129)!!
```
cd ..  # go back to the lib folder
python setup.py build develop
```
**2) Make a symlink back to this repository.**
```
ln -s /path/to/faster-rcnn.pytorch/lib /path/to/swapnet-repo/lib
```
Note: symlinks on Linux tend to work best when you provide the full path.

# Dataset
The MPV dataset is using here. You can download from here: https://drive.google.com/drive/folders/1e3ThRpSj8j9PaCUw8IrqzKPDVJK_grcA



## (Optional) Create Your Own Dataset
It is hard for having same dataset: Running Human parsing and pose estimation for preprocessing data.

### Preprocessing


# Training

Train progress can be viewed by opening `localhost:8097` in your web browser.

1) Train stage I
```
python train.py --name deep_fashion/warp --model warp --dataroot data/deep_fashion
```


2) Train Stage II
```

```


# Inference

```
python inference.py --checkpoint checkpoints/deep_fashion \
  --dataroot data/deep_fashion \
  --shuffle_data True
```



```
python inference.py --checkpoint checkpoints/deep_fashion \
  --cloth_dir [SOURCE] --texture_dir [SOURCE] --body_dir [TARGET]
```
Where SOURCE contains the clothing you want to transfer, and TARGET contains the person to place clothing on.

# Comparisons to Original MGVTON
### Similarities
- Stage I
  - [x] Test
- Stage II
  - [x] Test
- Stage III: Refinement render
  - [x] Test

### Differences


### TODO:
- [ ] Implement Stage I: generator and Discriminator
- [ ] Implement Geometric matching module GMM(body shape, target cloth mask) --> warped cloth mask
- [ ] Implement Geometric Matcher GMatcher(References parsing) --> Synthesys Parsing
- [ ] Implement Warp-GAN: Generator and Discriminator
- [ ] Implementation of refinement render
- [ ] Add regularize to GMM and GMatcher
- [ ] DeformableGAN --> Decomposed DeformableGAN

# What's Next?
### Stage I
- [ ] Check generated parsing performance, especially compare with the paper and others
- [ ] Add bottoms, with sample short and long pants or short and long skirts.
- [ ] Increase the weights of gan loss and check the results
- [ ] Think to change the input and network structures. For example. Mask input instead of color cloth input. And the effects of residual network. by comparing the results without it.
- [ ] Change the weight on loss of difference part of clothes. 
+ boundary focus more
+ base on average area all the dataset (statistical) ==> weight on loss of each human part label
+ soft weight, some clothes on same body part effect the weight on loss ==> skirt and pant on same bottom...

### Stage II


# Credits

