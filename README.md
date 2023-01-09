# Raga-Classifier

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GaneshTS06/Raga-Classifier/HEAD?labpath=main.ipynb)

This work aims to classify audio input according to the raaga being played. The  fixed ratio  of all the swaras in a raaga relative to the shadja is used. The Discrete Fourier Transform(DFT) and some filtering techniques are used to remove the noise in the audio. The Fisher-Jenks Natural Breaks Optimization Algorithm is used for clustering the frequencies, after which the highest frequency in a cluster is considered as dominant frequency, and all dominant frequencies are then normalised with lowest dominant frequency. The Euclidean Distance between thus obtained ratios and database is compared, and the lowest distance value entry is identified as raaga.
