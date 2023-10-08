# Super Mario Bros
`pip install gym_super_mario_bros==7.3.0 nes.py==8.1.8`

# Setuptools
`pip install setuptools==59.1.0`

# Gym
`pip install gym==0.21.0`
- stable-baselines3 doesn't support gym v0.26

# Trouble shooting
- 'EntryPoints' object has no attribute 'get' (https://github.com/InceptioResearch/itac/issues/1)
    `pip install importlib-metadata==4.0`
- ModuleNotFoundError: No module named 'cv2'
    `pip install opencv-python`
- ImportError: Trying to log data to tensorboard but tensorboard is not installed
    `pip install tensorboard`

# Stable Baselines
`pip install stable-baselines3==1.5.0`
- stable-baselines3 V2 doesn't support gym 
- stable-baselines3 V1.x doesn't support gym v0.26
- 
# PyTorch for Reinforcement Learning
For compatibility with gym version and stable-baselines3, choose the previous versions around v1.10.1
https://pytorch.org/get-started/previous-versions/

## For MacOS
`conda install pytorch::pytorch torchvision torchaudio -c pytorch`

## For Windows
### CUDA
`pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`


