# Optimal transport applied to infographics (color transportation problem)

Optimal transport applied to color transportation in the field of infographics / image processing.

## Abstract

The shortest path principle guides most decisions in life and sciences and therefore, optimization problems have came to the fore. The goal of Optimal Transport as a mathematical gem at the interface between probability, analysis and optimization is to find the least costly transport. This work reviews this field with a bias toward numerical methods and their applications in computer graphics, and sheds lights on the impact of the given distance on the final result.

![alt text](https://github.com/AdrienC21/optimal-transport-color-transportation/blob/main/images/results/optimal_transport.jpg?raw=true)
## Article

A short [article](https://github.com/AdrienC21/optimal-transport-color-transportation/blob/main/ressources/article.pdf) has been written (**in french**) to sum up the ideas behind and the key results.

<p align="center">
  <img src="https://github.com/AdrienC21/optimal-transport-color-transportation/blob/main/ressources/article-1.png" width="700">
  <img src="https://github.com/AdrienC21/optimal-transport-color-transportation/blob/main/ressources/article-2.png" width="700">
  <img src="https://github.com/AdrienC21/optimal-transport-color-transportation/blob/main/ressources/article-3.png" width="700">
  <img src="https://github.com/AdrienC21/optimal-transport-color-transportation/blob/main/ressources/article-4.png" width="700">
</p>

## Installation

Clone this repository :

```bash
git clone https://github.com/AdrienC21/optimal-transport-color-transportation.git
```

Make sure the following packages are installed. If not, type in a python console :

```python
pip install --upgrade pip
pip install setuptools
pip install --upgrade setuptools --ignore-installed

pip install numpy
pip install matplotlib
pip install scipy
pip install cython
pip install POT
pip install colour-science
pip install colour-science[optional]
pip install colour-science[plotting]
pip install colour-science[tests]
pip install colour-science[docs]
pip install colour-science[development]

pip install pymanopt autograd
```

## How to use

Edit in parameters.py the following lines :

```python
# name of the source image (the one that will change color)
imageSourceName = "bluebutterfly.jpg"
# name of the target image (colors of this one will be transported
# onto the source image)
imageTargetName = "pinkfield.jpeg"
nbpixels = 1000  # number of pixels that will be randomly chosen
```

Run run_optimal_transport.py to apply transport optimal algorithms.

**WARNING :** The complexity of our methods are O(n*m1 + n**4) where n is equal to nbpixels and m1 is the number of pixels in the source image, which means that nbpixels is a really sensitive parameter regarding the running time.

## Bibliography

[1] Cohen Scott : Finding color and shape patterns in images, Thèse, InfoLab Stanford, Chapitre 4, Mai 1999

[2] Gabriel Peyré : Le transport optimal: de Gaspard Monge à la science des données,Conférence, 2018

[3] Ferradans, S., Papadakis, N., Peyre, G., & Aujol, J. F. : (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882

[4] M. Perrot, N. Courty, R. Flamary, A. Habrard : "Mapping estimation for discrete optimal transport", Neural Information Processing Systems (NIPS), 2016

[5] Gabriel Peyré : Convex Optimization, note de cours

[6] Lindbloom Bruce : RGB/XYZ Matrices : [http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html](http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html), consultation : Nov 2018

[7] Gabriel Peyré : Computational Optimal Transport, Mars 2018

[8] Yann Brenier, Thierry Viéville : «La brouette de Monge ou le transport optimal » - Images des Mathématiques, CNRS, 2012

## License

[MIT](https://choosealicense.com/licenses/mit/)
