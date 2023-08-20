# Installation Commands
Note: Running on Windows 10 and using Anaconda for virtual environments

- Using Visual Studio, I downloaded the "Desktop Development with C++" workload
- Then, by launching VS Code from Anaconda, I ran these commands in the VS Code Terminal
- `conda create -n mario`
- `conda activate mario`
- `conda install pip`
- `conda install -c conda-forge cxx-compiler`
- `pip install nes-py`
- `pip3 install gym-super-mario-bros`

## Running the Test Code
The test code worked, but I did notice that in the code given, on line 5, it should be `'SuperMarioBros-v0'` not `’SuperMarioBros-v0’` for the code to work. Probably some sort of PDF export error? (I'm sure you would've picked up on this just adding this in for future reference)

