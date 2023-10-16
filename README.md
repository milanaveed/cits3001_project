# Environment Setup
Note: This Agent was setup on Windows 10, using Anaconda for virtual environments.

## Installation Steps
- Using Visual Studio, download the "Desktop Development with C++" workload. [Here is a link to a guide on how to do this](https://learn.microsoft.com/en-us/visualstudio/install/modify-visual-studio?view=vs-2022).
- In Anaconda Navigator, create a new environemnt "mario" that uses Python v3.7.16
- Launch VS Code from the Anaconda Navigator.
- Run these commands in the terminal:
- `conda activate mario`
- `conda install pip`
- `conda install -c conda-forge cxx-compiler`
- `pip install nes-py`
- `pip3 install gym-super-mario-bros`

## List of packages
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



