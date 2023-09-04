# For Windows 10
## Installation commands
Note: Running on Windows 10 and using Anaconda for virtual environments

- Using Visual Studio, I downloaded the "Desktop Development with C++" workload
- Then, by launching VS Code from Anaconda, I ran these commands in the VS Code Terminal
- `conda create -n mario` - I found that using this will crate an environemnt in the latest version of Python. If I'm not mistaken, in the latest Tutorial, the lecturer said that it wont work for any version other than 3.6-3.8. So, a better way to go about this is to make the environment in the Anaconda Navigator (I chose 3.7.16), and launch VS Code from Anaconda Navigator with this new environment selected. So far, there aren't any differences in how the code ran, but maybe down the line if we keep using the latest version of Python it might not work.
- `conda activate mario`
- `conda install pip`
- `conda install -c conda-forge cxx-compiler`
- `pip install nes-py`
- `pip3 install gym-super-mario-bros`


## Running the Test Code
The test code worked, but I did notice that in the code given, on line 5, it should be `'SuperMarioBros-v0'` not `’SuperMarioBros-v0’` for the code to work. Probably some sort of PDF export error? (I'm sure you would've picked up on this just adding this in for future reference).

Another thing I noticed was that the gaem was running abnormally fast in comparison with David's example in the Tutorial where he showed us. There might be something wrong with the emulator speed (I've asked about this in help3001 [here](https://secure.csse.uwa.edu.au/run/help3001?p=np&opt=B73&year=2023).

# For Mac OS
## Installation and Running with the Correct Python Interpreter
Create the environment in Anaconda by choosing python 3.7.16, and launch VS code from Anaconda.
- `conda activate mario`
- `pip install gym-super-mario-bros` 

## Troubleshooting
- use `which python` to check python interpreter
- if not the desired python interpreter, `echo $PATH` to inspect the python interpreter path, and `export PATH="$HOME/anaconda/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:$PATH"` to make sure the anaconda python interpreter have precedence
- if any module not found, `pip install <module name>`