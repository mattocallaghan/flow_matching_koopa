# Koopman-Lifted Flow-Matching-Posterior-Estimation


Towards fast, accurate, interperable and flexible simulation based inference using koopman-lifted flow matching objectives.


This repository builds on the Flow Matching Posterior Estimation branch of [dingo](https://github.com/dingo-gw/dingo), based on the NeurIPS-2023 paper [Flow Matching for Scalable 
Simulation Based Inference](https://neurips.cc/virtual/2023/poster/72395) package for the
implementation of the base methods for flow matching.


# Requirements
Install `dingo`.

```sh
cd dingo
pip install -e ."[dev]"
```

Install the [sbibm](https://github.com/sbi-benchmark/sbibm) package for the 
benchmark experiments.

```sh
pip install sbibm
```

# Experiments

## SBI Benchmark 

Training and evaluation scripts available in `./sbi-benchmark`.





Please also refer to the documentation: https://dingo-gw.readthedocs.io.


# References

```bibtex
@inproceedings{wildberger2023flow,
    title={Flow Matching for Scalable Simulation-Based Inference},
    author={Jonas Bernhard Wildberger and Maximilian Dax and Simon Buchholz and Stephen R Green and Jakob H. Macke and Bernhard Sch{\"o}lkopf},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=D2cS6SoYlP},
    eprint={2305.17161},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
}
@article{Dax:2021tsq,
    author = {Dax, Maximilian and Green, Stephen R. and Gair, Jonathan and Macke, Jakob H. and Buonanno, Alessandra and Sch\"olkopf, Bernhard},
    title = "{Real-Time Gravitational Wave Science with Neural Posterior Estimation}",
    eprint = "2106.12594",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "LIGO-P2100223",
    doi = "10.1103/PhysRevLett.127.241103",
    journal = "Phys. Rev. Lett.",
    volume = "127",
    number = "24",
    pages = "241103",
    year = "2021"
}
```



