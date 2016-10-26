# machineLearning

## Euler
bsub -n 2 -R "rusage[mem=8000]" python -singleCompThread -r compute_all_peaks.py 

## Dependencies

1. nibabel
2. numpy
3. Do we need scypy?
4. Do we needscikit-learn ?
5. Do we need pandas ?

Install on linux systems (and brutus/euler servers)

module load python/2.7.6

python get-pip.py --user

python -m pip install --user nibabel

python -m pip install --user numpy

python -m pip install --user matplotlib

python -m pip install --user scipy

pip install --user --install-option="--prefix=" -U scikit-learn



## Main TODO 
We could try to use the Issues tab instead.

## Ideas (Barth tu peux t'exprimer ici)

Commandes console pour télécharger les packages qui seront à priori nécessaires :
pip install nibabel
pip install matplotlib
pip install numpy
pip install -U scikit-learn
pip install scipy

Peut-être se mettre d'accord dès maintenant sur une syntaxe commune du style :
import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import Lasso
