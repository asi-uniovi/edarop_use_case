# Edarop use case

This repository contains the code for an example use case of
[Edarop](https://github.com/asi-uniovi/edarop).

To run the example, you need to install Edarop and the other dependencies of
this project. The easiest way to do so is to create a virtual environment with
`pip` and install the dependencies from `requirements.txt`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

In order to obtain the data from Edarop and Malloovia, you have to run:

```bash
python use_case_edarop.py
python use_case_malloovia.py
```

The results will be stored in the files `sols_edarop.p` and `sols_malloovia.p`
respectively.

To obtain the figures and tables of the paper, use the notebook `Paper.ipynb`.
