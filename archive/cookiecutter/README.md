# cookiecutter

[![package](https://github.com/lingjie00/template/actions/workflows/project-actions.yml/badge.svg)](https://github.com/lingjie00/template/actions/workflows/project-actions.yml)
[![Docker](https://github.com/lingjie00/template/actions/workflows/docker-actions.yml/badge.svg)](https://github.com/lingjie00/template/actions/workflows/docker-actions.yml)

# Project overview

testing cookiecutter

# Methodology

Please refer to /docs

# Code, Notebook examples

Please refer to /notebooks

# Datasets

Please upload data to /data

Take note of privacy and copyrights.

# Statis resources

For storing static resources such as images, graphs etc

Please upload to /res

# Package

Please upload to /src

# Local development

install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
and create new virtual environment

create new environment, replace ```cookiecutter``` with the env name
```bash
conda create -n cookiecutter python==3.9
```

activate the new environment
```bash
conda activate cookiecutter
```

install essential packages and this repo package
```bash
pip install -r requirements.txt
```


# Reprodue code

Please include requirements.txt
```python
conda list --export > requirements.txt
```

