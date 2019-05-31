# Transformation Autoregressive Networks (TANs)
### A state of the art density estimator

This is the official repository for the [2018 ICML article](http://proceedings.mlr.press/v80/oliva18a.html).
```
@inproceedings{oliva2018transformation,
  title={Transformation Autoregressive Networks},
  author={Oliva, Junier and Dubey, Avinava and Zaheer, Manzil and Poczos, Barnabas and Salakhutdinov, Ruslan and Xing, Eric and Schneider, Jeff},
  booktitle={International Conference on Machine Learning},
  pages={3895--3904},
  year={2018}
}
```

**TLDR**: Combining novel change of variables jointly with autoregressive conditional 1d-mixture-models yields an extremely powerful density estimator on general real valued data.
A year later, TANs are still leading on several benchmark datasets.
Watch the 5-minute video clip below for a quick summary:

[![video](https://i.imgur.com/eLC2oAm.png "Spotlight")](https://www.facebook.com/icml.imls/videos/432151720632682/?t=2620)

***
## Results
Below we compile some of the state of the art mean test log likelihoods on several benchmark dataset introduced in the [MAF paper](http://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation). Standard errors are reported when available. **Please report the published results below when reporting TANs performance on these datasets.** *Please do not report unpublished early results from an ICLR submission as done in * [[1]](https://openreview.net/pdf?id=rJxgknCcK7) [[2]](https://openreview.net/pdf?id=rJxgknCcK7).

| Method        | POWER           | GAS  | HEPMASS  | MINIBOONE  |BSDS300
| ------------- |
| TANs [[*2018*]](http://proceedings.mlr.press/v80/oliva18a.html)   | *0.60 ± 0.01* | **12.06 ± 0.02** | **−13.78 ± 0.02** | −11.01 ± 0.48 | **159.80 ± 0.07**
| SOS [[*2019*]](https://arxiv.org/pdf/1905.02325.pdf)          | *0.60 ± 0.01* | 11.99 ± 0.41     | -15.15 ± 0.10 | -8.90 ± 0.11 | 157.48 ± 0.41
| FFJORD [[*2019*]](https://openreview.net/pdf?id=rJxgknCcK7)       | 0.46 | 8.59 | -14.92 | -10.43 | 157.40 |
| NAF DDSF (5) [[*2018*]](https://openreview.net/pdf?id=rJxgknCcK7) | *0.62 ± 0.01* | 11.91 ± 0.13   | -15.09 ± 0.40 | **-8.86 ± 0.15**      | 157.73 ± 0.04
| NAF DDSF (10) [[*2018*]](https://openreview.net/pdf?id=rJxgknCcK7) | *0.60 ± 0.02* | 11.96 ± 0.33     | -15.32 ± 0.23 | -9.01 ± 0.01      | 157.43 ± 0.30
| MAF (5) [[*2017*]](http://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation)      | 0.14 ± 0.01 | 9.07 ± 0.02  | -17.70 ± 0.02 | -11.75 ± 0.44 | 155.69 ± 0.28
| MAF (10) [[*2017*]](http://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation)      | 0.24 ± 0.01 | 10.08 ± 0.02 | -17.73 ± 0.02 | -12.24 ± 0.45 | 154.93 ± 0.28
| MAF MoG (5) [[*2017*]](http://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation)   | 0.30 ± 0.01 | 9.59 ± 0.02  | -17.39 ± 0.02 | -11.68 ± 0.44 | 156.36 ± 0.28
| Real NVP (5) [[*2016*]](https://arxiv.org/abs/1605.08803)\* | -0.02 ± 0.01| 4.78 ± 1.80  | -19.62 ± 0.02 | -13.55 ± 0.49 | 152.97 ± 0.28
| Real NVP (10) [[*2016*]](https://arxiv.org/abs/1605.08803)\* | 0.17 ± 0.01 | 8.33 ± 0.14  | -18.71 ± 0.02 | -13.84 ± 0.52 | 153.28 ± 1.78
| MADE [[*2015*]](http://proceedings.mlr.press/v37/germain15.pdf)\*         |-3.08 ± 0.03 | 3.56 ± 0.04  | -20.98 ± 0.02 | -15.59 ± 0.50 | 148.85 ± 0.28
| MADE MoG [[*2015*]](http://proceedings.mlr.press/v37/germain15.pdf)\*      | 0.40 ± 0.01 | 8.47 ± 0.02  | -15.15 ± 0.02 | -12.27 ± 0.47 | 153.71 ± 0.28

\*As reported in the [MAF paper](http://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation).

To train model on exactly the same data as the [MAF paper](http://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation), first download and save data in pickle formats:
```python
import tan.demos.mafdata_demo as mdemo
# Will save pickle files in ~/data/tan by default
# Assumes you are in outermost repo directory; i.e. can see ./tan/external_maf
mdemo.download_and_make_data()
```
After, train a model on a specified dataset by running
```python
# Returns a dictionary of result stats
res = mdemo.main(dataset=dname)
```
where `dname` is one of {`maf_power`, `maf_gas`, `maf_hepmass`, `maf_miniboone`, `maf_bsds`}.
A scatter plot of samples projected into the 2d space of first and last dimensions is also saved (by default in `~/data/tan/maf_*/`); for example for `maf_hepmass`:
[![hepmass](https://i.imgur.com/VWXasRy.png "scatter2d")](https://www.facebook.com/icml.imls/videos/432151720632682/?t=2620)

Note that the above validates across multiple random initializations (*open question: how to best initialize?*), if you wish to only train on one initialization, then run `res = mdemo.main(dataset=dname, ntrls=1)`.
