## LangDec

**LangDec** is a lightweight framework for experimenting with decoding algorithms in large language models.
It supports both standard decoding strategies (e.g., greedy, beam, sampling) and **guided decoding** using **Process Reward Models (PRMs)**.
The framework is designed to facilitate research reproducibility and rapid prototyping.

### Key Features

* Unified interface for running multiple decoding strategies
* Support for PRM-guided step-wise scoring and branch selection
* Fully scriptable experiment execution with logging and checkpointing
* Example experiment script: `scripts/run_sc.sh`

---

## Installation

```bash
# Clone the repository
git clone <YOUR_LANGDEC_REPOSITORY_URL> langdec
cd langdec

# (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start

An example execution script is provided at:

```
scripts/run_sc.sh
```


---

## Acknowledgement

This project builds upon the open-source implementation of **VersaPRM** from UW-Madison Lee Lab.
We thank the authors for providing high-quality research code and open scientific contributions.

* VersaPRM Repository: [https://github.com/UW-Madison-Lee-Lab/VersaPRM](https://github.com/UW-Madison-Lee-Lab/VersaPRM)

If you use or reference this work, please cite accordingly.

---