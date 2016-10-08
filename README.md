# machineLearning

## Dependencies

1. nibabel
2. numpy
3. Do we need scypy?
4. Do we needscikit-learn ?
5. Do we need pandas ?

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
import matplotlib as plt
import sklearn as sk
from sklearn.linear_model import Lasso

Pour visualiser des images (ne marche pas chez moi (Barthe)) :
http://nipy.org/nibabel/coordinate_systems.html

We can load up the EPI image to get the image data array:
>>> import nibabel as nib
>>> epi_img = nib.load('downloads/someones_epi.nii.gz')
>>> epi_img_data = epi_img.get_data()
>>> epi_img_data.shape
(53, 61, 33)

Then we have a look at slices over the first, second and third dimensions of the array.
import matplotlib.pyplot as plt
def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = epi_img_data[26, :, :]
slice_1 = epi_img_data[:, 30, :]
slice_2 = epi_img_data[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")  
