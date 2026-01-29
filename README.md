# Confidence-Adaptive Lipschitz-Managed Control Barrier Functions (CALM-CBFs)
Git repository for Confidence-Adaptive Lipschitz-Managed Control Barrier Functions (CALM-CBFs).

## Install
Run the following commands:
```bash
# create directory
mkdir -p ~/calmcbf
cd ~/calmcbf

# create environment
pyenv install -s 3.10.8
pyenv virtualenv 3.10.8 calmcbf
pyenv activate calmcbf

# install forked cbfpy
git clone https://github.com/PietDol/cbfpy.git
cd cbfpy
pip install -e .
cd ..
python -c "import cbfpy; print('cbfpy import OK')"

# clone calm cbf repo
git clone https://github.com/PietDol/CALM-CBF.git
cd CALM-CBF
pip install -r requirements.txt
cd ..


# run the experiment script
cd CALM-CBF/source
python experiments.py
```

# 