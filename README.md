-   John Giampaolo (23395411)
-   Mila Zhang (22756463)

# Environment Setup for Rule-base Agent

Note: This Agent was setup on Windows 10, using Anaconda for virtual environments.

## Installation Steps

-   Using Visual Studio, download the "Desktop Development with C++" workload. [Here is a link to a guide on how to do this](https://learn.microsoft.com/en-us/visualstudio/install/modify-visual-studio?view=vs-2022).
-   In Anaconda Navigator, create a new environemnt "mario" that uses Python v3.7.16
-   Launch VS Code from the Anaconda Navigator.
-   Run these commands in the terminal:
-   `conda activate mario`
-   `conda install pip`
-   `conda install -c conda-forge cxx-compiler`
-   `pip install nes-py`
-   `pip3 install gym-super-mario-bros`

## List of Packages

In case something went wrong, here is a list of packages used by the environemnt we set up. (Found by running command `conda list`):

```
# Name                    Version                   Build  Channel
ca-certificates           2023.05.30           haa95532_0
certifi                   2022.12.7        py37haa95532_0
cloudpickle               2.2.1                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi
git                       2.40.1               haa95532_1
gym                       0.26.2                   pypi_0    pypi
gym-notices               0.0.8                    pypi_0    pypi
gym-super-mario-bros      7.4.0                    pypi_0    pypi
importlib-metadata        6.7.0                    pypi_0    pypi
nes-py                    8.2.1                    pypi_0    pypi
numpy                     1.21.6                   pypi_0    pypi
opencv-python             4.8.0.76                 pypi_0    pypi
openssl                   1.1.1v               h2bbff1b_0
pip                       22.3.1           py37haa95532_0
pyglet                    1.5.21                   pypi_0    pypi
python                    3.7.16               h6244533_0
setuptools                65.6.3           py37haa95532_0
sqlite                    3.41.2               h2bbff1b_0
tqdm                      4.66.1                   pypi_0    pypi
typing-extensions         4.7.1                    pypi_0    pypi
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
wheel                     0.38.4           py37haa95532_0
wincertstore              0.2              py37haa95532_2
zipp                      3.15.0                   pypi_0    pypi
```

# Environment Setup for Reinforcement Learning Agent

## Python Version

`3.7.16`

## Super Mario Bros

`pip install gym_super_mario_bros==7.3.0 nes.py==8.1.8`

## Setuptools

`pip install setuptools==59.1.0`

## Gym

`pip install gym==0.21.0`

-   stable-baselines3 doesn't support gym v0.26

## Troubleshooting

-   'EntryPoints' object has no attribute 'get' (https://github.com/InceptioResearch/itac/issues/1)
    `pip install importlib-metadata==4.0`
-   ModuleNotFoundError: No module named 'cv2'
    `pip install opencv-python`
-   ImportError: Trying to log data to tensorboard but tensorboard is not installed
    `pip install tensorboard`

## Stable Baselines

`pip install stable-baselines3==1.5.0`

-   stable-baselines3 V2 doesn't support gym
-   stable-baselines3 V1.x doesn't support gym v0.26
-

## PyTorch for Reinforcement Learning

For compatibility with gym version and stable-baselines3, choose the previous versions around v1.10.1
https://pytorch.org/get-started/previous-versions/

#### For MacOS

`conda install pytorch::pytorch torchvision torchaudio -c pytorch`

#### For Windows

###### CUDA

`pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

## Packages in the Environment (conda list)

```
## Name                    Version                   Build  Channel
absl-py                   2.0.0                    pypi_0    pypi
ca-certificates           2023.08.22           haa95532_0
cachetools                5.3.1                    pypi_0    pypi
certifi                   2022.12.7        py37haa95532_0
charset-normalizer        3.3.0                    pypi_0    pypi
cloudpickle               2.2.1                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi
cycler                    0.11.0                   pypi_0    pypi
fonttools                 4.38.0                   pypi_0    pypi
google-auth               2.23.2                   pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
grpcio                    1.59.0                   pypi_0    pypi
gym                       0.21.0                   pypi_0    pypi
gym-notices               0.0.8                    pypi_0    pypi
gym-super-mario-bros      7.3.0                    pypi_0    pypi
idna                      3.4                      pypi_0    pypi
importlib-metadata        4.0.0                    pypi_0    pypi
kiwisolver                1.4.5                    pypi_0    pypi
markdown                  3.4.4                    pypi_0    pypi
markupsafe                2.1.3                    pypi_0    pypi
matplotlib                3.5.3                    pypi_0    pypi
memory-profiler           0.61.0                   pypi_0    pypi
nes-py                    8.1.8                    pypi_0    pypi
numpy                     1.21.6                   pypi_0    pypi
oauthlib                  3.2.2                    pypi_0    pypi
opencv-python             4.8.1.78                 pypi_0    pypi
openssl                   1.1.1w               h2bbff1b_0
packaging                 23.2                     pypi_0    pypi
pandas                    1.3.5                    pypi_0    pypi
pillow                    9.5.0                    pypi_0    pypi
pip                       23.2.1                   pypi_0    pypi
protobuf                  3.20.3                   pypi_0    pypi
psutil                    5.9.6                    pypi_0    pypi
pyasn1                    0.5.0                    pypi_0    pypi
pyasn1-modules            0.3.0                    pypi_0    pypi
pyglet                    1.5.11                   pypi_0    pypi
pyparsing                 3.1.1                    pypi_0    pypi
python                    3.7.16               h6244533_0
python-dateutil           2.8.2                    pypi_0    pypi
pytz                      2023.3.post1             pypi_0    pypi
requests                  2.31.0                   pypi_0    pypi
requests-oauthlib         1.3.1                    pypi_0    pypi
rsa                       4.9                      pypi_0    pypi
setuptools                59.1.0                   pypi_0    pypi
six                       1.16.0                   pypi_0    pypi
sqlite                    3.41.2               h2bbff1b_0
stable-baselines3         1.5.0                    pypi_0    pypi
tensorboard               2.11.2                   pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
torch                     1.10.1+cu113             pypi_0    pypi
torchaudio                0.10.1+cu113             pypi_0    pypi
torchvision               0.11.2+cu113             pypi_0    pypi
tqdm                      4.66.1                   pypi_0    pypi
typing-extensions         4.7.1                    pypi_0    pypi
urllib3                   2.0.6                    pypi_0    pypi
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
werkzeug                  2.2.3                    pypi_0    pypi
wheel                     0.38.4           py37haa95532_0
wincertstore              0.2              py37haa95532_2
zipp                      3.15.0                   pypi_0    pypi
```
