# Gym
pip install gym==v0.21.0
- stable-baselines3 doesn't support gym v0.26

# Super Mario Bros usually 7.4
pip install gym_super_mario_bros==v7.3.0

# PyTorch
For compatibility with gym version and stable-baselines3, choose the previous versions around v1.10.1
https://pytorch.org/get-started/previous-versions/

## For MacOS
conda install pytorch::pytorch torchvision torchaudio -c pytorch

## For Windows
### CUDA
- pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f

# Stable Baselines
pip install stable-baselines3==1.5.0

- stable-baselines3 V2 doesn't support gym 
- stable-baselines3 V1.x doesn't support gym v0.26