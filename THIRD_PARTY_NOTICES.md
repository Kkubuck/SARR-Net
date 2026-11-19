# Third-party notices

SARR-Net is released under the [Apache License 2.0](LICENSE). The components below keep their original notices and licenses; the source files name their upstream origin.

| Component | Files | Source | License |
|:--|:--|:--|:--|
| SMT backbone | `sarrnet/models/smt.py` | [AFeng-x/SMT](https://github.com/AFeng-x/SMT) | MIT, [`licenses/SMT-MIT.txt`](licenses/SMT-MIT.txt) |
| Saliency metric algorithms | `sarrnet/evaluation/metrics.py` | [lartpang/PySODMetrics](https://github.com/lartpang/PySODMetrics) | MIT, [`licenses/PySODMetrics-MIT.txt`](licenses/PySODMetrics-MIT.txt) |

The SMT file passes the number of input bands to the convolutional stem and returns the four stage feature maps; the profiling, registry and transform helpers are removed.

`assets/overview.png`, `assets/qualitative.png` and `assets/error_maps.png` are taken from Figures 2, 5 and 6 of the authors' article accepted to *IEEE Transactions on Multimedia*. MCOD example images remain covered by the dataset's terms, including its CC BY-NC-ND 4.0 license. Apart from the examples embedded in these figures, dataset images are not distributed here. Model weights are not included.
