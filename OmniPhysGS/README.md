# [ICLR 2025] OmniPhysGS

<h4 align="center">

OmniPhysGS: 3D Constitutive Gaussians for General Physics-based Dynamics Generation

[Yuchen Lin](https://wgsxm.github.io), [Chenguo Lin](https://chenguolin.github.io), [Jianjin Xu](https://atlantixjj.github.io/), [Yadong Mu](http://www.muyadong.com)

[![arXiv](https://img.shields.io/badge/arXiv-2501.18982-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.18982)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://wgsxm.github.io/projects/omniphysgs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

<p>
    <img width="90%" alt="pipeline", src="./assets/teaser.png">
</p>

</h4>

This repository contains the official implementation of the paper: [OmniPhysGS: 3D Constitutive Gaussians for General Physics-based Dynamics Generation](https://wgsxm.github.io/projects/omniphysgs/), which is accepted by ICLR 2025.
OmniPhysGS is a novel framework for general physics-based 3D dynamic scene synthesis, which can automatically and flexibly model various materials with domain-expert constitutive models in a physics-guided network. 
Here is our [Project Page](https://wgsxm.github.io/projects/omniphysgs).

Feel free to contact me (linyuchen@stu.pku.edu.cn) or open an issue if you have any questions or suggestions.


## 📢 News
- **2025-03-26**: Upload several trained Gaussian models and the corresponding config files to [this link](https://drive.google.com/drive/folders/1B_CWR8CnjHVhOt_CmSvenlI7IirsaVVH?usp=sharing).
- **2025-03-19**: A clean version of our PyTorch MPM solver is released [here](https://github.com/wgsxm/MPM-PyTorch).
- **2025-02-03**: The source code and preprocessed dataset are released.
- **2025-01-22**: OmniPhysGS is accepted by ICLR 2025.

## 🔧 Installation
Our code uses [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) as an important submodule. If problem occurs during the installation, please refer to the official repository for more details. 
### Clone the repository and submodules
```bash
git clone --recurse-submodules https://github.com/wgsxm/omniphysgs.git
cd omniphysgs
```
### Change the version of Gaussian Splatting code (optional)
We use a specific version of Gaussian Splatting code in our project. Later versions may cause compatibility issues. Please check the hash of the Gaussian Splatting repository in the `third_party/gaussian-splatting` folder and make sure it is as follows: 
```bash
cd third_party/gaussian-splatting
git checkout 472689c0dc70417448fb451bf529ae532d32c095
cd ../..
```
### Setup the environment
```bash
conda create -n omniphysgs python=3.11.9
conda activate omniphysgs
bash settings/setup.sh
```
These commands should install all the dependencies required to run the code. 

## 🌎 Environment
You may need to modify the specific version of `torch` in `settings/setup.sh` according to your CUDA version. 
We provide our environment configuration here for reference. We recommend using the same environment to reproduce our results. See `settings/requirements.txt` for more details.
- `Ubuntu 22.04.1 LTS`
- `Python 3.11.9`
- `CUDA 11.8`
- `torch==2.0.1`
- `warp-lang==0.6.1`
- `diffusers==0.30.3`

## 📊 Dataset

We provide an example data in the `dataset` folder and its corresonding configuration file in the `configs` folder.
The dataset is organized as follows (the same as the **training results** of [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)):

```
omniphysgs
├── dataset
│   ├── bear
│   │   ├── point_cloud
│   │   │   ├── iteration_30000
│   │   │   │   ├── point_cloud.ply
│   │   ├── cameras.json
│   ├── ... # other scenes
|   |   ├── point_cloud
|   |   │   ├── iteration_*  # the number of iterations depends on the training process
|   |   │   │   ├── point_cloud.ply
|   |   ├── cameras.json
```
More data will be released soon in this Google Drive [link](https://drive.google.com/drive/folders/1B_CWR8CnjHVhOt_CmSvenlI7IirsaVVH?usp=sharing).

## 🚀 Usage
Note that given the feature of SDS guidance, loss cannot explicitly reflect the quality of the generated results. We recommend checking the intermediate video results in the output folder. 

### Training
We provide two training config files in the `configs` folder for the same example scene (a bear). 
They are almost identical except for the `prompt` field. To train the model, simply run: 
```bash
python main.py --config configs/bear_sand.yaml --tag bear_sand
```
The argument `--tag` is used to specify the name of the subfolder in the output directory. 
In this case, the output will be saved in `outputs/bear_rubber`. 
If not specified, the default tag will be a timestamp. 

If you want to specify the GPU device, you can add the `--gpu` argument:
```bash
python main.py --config configs/bear_sand.yaml --tag bear_sand --gpu 0
```

### Inference
After training, you can generate the dynamic scene by running:
```bash
python main.py --config configs/bear_sand.yaml --tag bear_sand --test
```
This will recover the trained physics-guided network and generate the dynamic scene. 
Gradient is disabled during inference to speed up the process. 
By changing `boundary_condition` in the config file, 
you can generate different dynamic scenes with the same learned material. 

### Results
Using the provided config files, you can train and generate the results similar to the following videos: 

<div style="display: flex; justify-content: space-around;">
  <div style="text-align: center;">
    <img src="./assets/bear_rubber.gif" alt="bear_rubber" width="400">
    <p><i>"A rubber bear bouncing on a surface"</i></p>
  </div>
  <div style="text-align: center;">
    <img src="./assets/bear_sand.gif" alt="bear_sand" width="400">
    <p><i>"A sand bear collapsing"</i></p>
  </div>
</div>



## 😊 Acknowledgement
We would like to thank the authors of [PhysGaussian](https://xpandora.github.io/PhysGaussian/), [PhysDreamer](https://physdreamer.github.io/), [Physics3D](https://liuff19.github.io/Physics3D/), [DreamPhysics](https://github.com/tyhuang0428/DreamPhysics), and [NCLaw](https://sites.google.com/view/nclaw) for their great work and generously providing source codes, which inspired our work and helped us a lot in the implementation.


## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@inproceedings{
  lin2025omniphysgs,
  title={OmniPhys{GS}: 3D Constitutive Gaussians for General Physics-Based Dynamics Generation},
  author={Yuchen Lin and Chenguo Lin and Jianjin Xu and Yadong MU},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=9HZtP6I5lv}
}
```
