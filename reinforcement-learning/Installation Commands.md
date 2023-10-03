
# PyTorch
https://pytorch.org/get-started/previous-versions/

## For MacOS
conda install pytorch::pytorch torchvision torchaudio -c pytorch

## For Windows
### CUDA 11.3
- pip3 install torch torchvision torchaudio --index-url 
- conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forg

# Stable Baselines
pip install stable-baselines3[extra]

- stable-baselines3 v2 doesn't support gym 
- stable-baselines3 v1.x doesn't support gym v0.26