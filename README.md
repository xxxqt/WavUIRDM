##Official implementation of the paper:  
##WavUIRDM: Wavelet-based Conditional Residual Denoising Diffusion Model for Universal Image Restoration  

## 🔗 Paper
Coming soon...

## Environment
* Python 3.79
* Pytorch 1.12

### Create a virtual environment and activate it.
```
conda create -n env python=3.7
conda activate env
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
```

## Dataset
Our dataset setup follows the same structure as [DiffUIR](https://github.com/iSEE-Laboratory/DiffUIR).

## Train and Test 
```
python train.py
```
Notably, change the 'task' id  to your task, low-light enhancement for 'light_only', deblur for 'blur', dehaze for 'fog', derain for 'rain', desnow for 'snow' 
```
python test.py
```

## 🙏 Acknowledgements
This project builds upon the excellent work of the following repositories:
- [DiffUIR]([https://github.com/user/repo-name](https://github.com/iSEE-Laboratory/DiffUIR)): Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model
We sincerely thank the authors for sharing their code and ideas.
