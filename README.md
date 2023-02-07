# Seismic amplitude response to internal heterogeneity of mass-transport deposits

Authors: Jonathan Ford (jford@ogs.it), Angelo Camerlenghi, Francesca Zolezzi and Marilena Calarco

## Abstract
Compared to unfailed sediments, mass-transport deposits are often characterised by a low-amplitude response in single-channel seismic reflection images. This ‘acoustic transparency’ amplitude signature is widely used to delineate mass-transport deposits and is conventionally interpreted as a lack of coherent internal reflectivity due to a loss of preserved internal structure caused by mass-transport processes. In this study we examine the variation in the single-channel seismic response with changing heterogeneity using synthetic 2-D elastic seismic modelling. We model the internal structure of mass-transport deposits as a two-component random medium, using the lateral correlation length (a_x) as a proxy for the degree of internal deformation, whilst maintaining approximately constant internal reflectivity with increasing deformation. For a controlled single-source synthetic model a reduction in observed amplitude with reduced a_x is consistently observed across a range of vertical correlation lengths (az ). For typical AUV sub-bottom profiler acquisition parameters, in a simulated mass-transport deposit with realistic elastic and geostatistical properties, we find that when a_x ≈ 1 m, recorded seismic amplitudes are, on average, reduced by ∼ 15% relative to unfailed sediments (a_x ≫ 103 m). We also observe that deformation significantly larger than core-scale (a_x > 0.1 m) can generate a significant amplitude decrease. These synthetic modelling results should discourage interpretation of the internal structure of mass-transport deposits based on seismic amplitudes alone, as ‘acoustically transparent’ mass-transport deposits may still preserve coherent, metre-scale internal structure. In addition, the minimum scale of heterogeneity required to produce a significant reduction in seismic amplitudes is likely much larger than the diameter of sediment cores, meaning that ‘acoustically transparent’ mass-transport deposits may still appear well-stratified and undeformed at core-scale.

## Reproducing the synthetic modelling results

### Set up the environment using Anaconda and pip
```
conda env create -f environment.yml
conda activate sbp
cd code
pip install -e .
export DEVITO_LANGUAGE=openmp
export OMP_NUM_THREADS=8
```

### Run synthetic modelling

1. Single-source synthetic example:
```
cd code/sbp_modelling/single_source
python forward_model.py
python analysis.py
```

2. Multi-source realistic synthetic example:
```
cd code/sbp_modelling/multi_source
python forward_model.py
python analysis.py
```

### Compile figures
```
cd code/figures
jupyter notebook
```

## Code and data archive

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7348120.svg)](https://doi.org/10.5281/zenodo.7348120)

Ford, Jonathan, Camerlenghi, Angelo, Zolezzi, Francesca, & Calarco, Marilena. (2022). Seismic amplitude response to internal heterogeneity of mass-transport deposits (revision_v2). Zenodo. [https://doi.org/10.5281/zenodo.7348120](https://doi.org/10.5281/zenodo.7348120)

## License

Code to reproduce the results and figures is made available under the BSD 3-clause license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE` for the full text.

The manuscript, figures, data and the results are made available under the [Creative Commons Attribution 4.0 International License][cc-by]. See `manuscript/LICENSE` for the full text.

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png