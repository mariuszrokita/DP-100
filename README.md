# Introduction
This repository contains various notebooks and resources that are part of my learning path towards the Microsoft [exam DP-100](https://docs.microsoft.com/en-us/learn/certifications/exams/dp-100).

# Getting started

## Virtual environment

```shell
conda create --name dp100 python=3.7 jupyter
conda activate dp100
```

## Install dependencies

```shell
pip install --upgrade pip
pip install -r requirements.txt
```

It is also worth installing the Jupyter Notebook Extensions:

```shell
pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install
```