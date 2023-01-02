# ECG-DeNoise

this is a personal experiment to explore the feasibility of neural networks as a replacements to conventional ECG analysis techniques to better classify beat and rhythim characterstics especially in non ideal signal quality.

It a 1D implementation of the U-Net model from the paper, [U-Net: Convolutional Networks for Biomedical Image Segmentation.](https://arxiv.org/abs/1505.04597v1) for ecg denoising.

The model was trained on the [Icentia11K: An Unsupervised Representation Learning Dataset for Arrhythmia Subtype Discovery](https://arxiv.org/abs/1910.09570) datset. [alternate dwonload link (from academictorrents).](https://academictorrents.com/details/af04abfe9a3c96b30e5dd029eb185e)

The data is labeled by automated systems and 24.38% of all labeled beats are undefined and the objective is to reduce this figure.

`TODO: compare against transformer based archtecture`

<!-- <h3>Aggregate statistics</h3> -->
## Dataset
---

### General Stats
---

|Statistic|# (units)|
|:---|:---|
|Number of patients|11,000|
|Number of labeled beats|2,774,054,987|
|Sample rate|250Hz|
|Segment size|2^{20}+1  = 1,048,577|
|Total number of segments|541,794 (not all patients have enough for 50 segments)|

### Beats
---
Beats are annotated at the R timepoint in the QRS complex.

|Symbol|Beat Description|Count|
|:---|:---|---:|
|N|Normal|2,061,141,216|
|S|ESSV (PAC): Premature or ectopic supraventricular beat, <br/>premature atrial contraction|19,346,728|
|V|ESV (PVC): Premature ventricular contraction, <br/>premature ventricular contraction|17,203,041|
|Q|Undefined: Unclassifiable beat|676,364,002|


### sample strips
---

![input sample 0][input_sample_plot_y]

<!-- [input_sample_plot_y]: cache/plots/sample_ds_out.png "sample y plot" -->
[input_sample_plot_y]: cache/plots/download%20(1).png "sample y plot"

![input sample 1](cache/plots/download.png)

![input sample 2](cache/plots/unded_normal%2000004,%2029,%20938225,%20940725.png)


