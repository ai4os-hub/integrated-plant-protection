# integrated_plant_protection

[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/integrated-plant-protection/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/integrated-plant-protection/job/main/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Usage

To launch it, first install the package then run [deepaas](https://github.com/ai4os/DEEPaaS):
```bash
git clone https://github.com/ai4os-hub/integrated-plant-protection
cd integrated-plant-protection
pip install -e .
deepaas-run --listen-ip 0.0.0.0
```
### Directly from Docker Hub

To run the Docker container directly from Docker Hub and start using the API simply run the following command:

```bash
$ docker run -ti -p 5000:5000 -p 6006:6006 -p 8888:8888 ai4oshub/integrated-plant-protection
```

This command will pull the Docker container from the Docker Hub [ai4oshub](https://hub.docker.com/u/ai4oshub/) repository and start the default command (`deepaas-run --listen-ip=0.0.0.0`).

### Building the container

If you want to build the container directly in your machine (because you want to modify the `Dockerfile` for instance) follow the instructions below:
```bash
git clone https://github.com/ai4os-hub/integrated-plant-protection
cd integrated-plant-protection
docker build -t ai4oshub/integrated-plant-protection .
docker run -ti -p 5000:5000 -p 6006:6006 -p 8888:8888 ai4oshub/integrated-plant-protection
```

## Project structure
```
│
├── Dockerfile             <- Describes main steps on integration of DEEPaaS API and
│                             integrated_plant_protection application in one Docker image
│
├── Jenkinsfile            <- Describes basic Jenkins CI/CD pipeline (see .sqa/)
│
├── LICENSE                <- License file
│
├── README.md              <- The top-level README for developers using this project.
│
├── .sqa/                  <- CI/CD configuration files
│
├── integrated_plant_protection    <- Source code for use in this project.
│   │
│   ├── __init__.py        <- Makes integrated_plant_protection a Python module
│   │
│   ├── api.py             <- Main script for the integration with DEEPaaS API
│   │
│   └── misc.py            <- Misc functions that were helpful accross projects
│
├── data/                  <- Folder to store the data
│
├── models/                <- Folder to store models
│
├── tests/                 <- Scripts to perfrom code testing
|
├── metadata.json          <- Defines information propagated to the AI4OS Hub
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
│                             generated with `pip freeze > requirements.txt`
├── requirements-test.txt  <- The requirements file for running code tests (see tests/ directory)
│
└── setup.py, setup.cfg    <- makes project pip installable (pip install -e .) so
                              integrated_plant_protection can be imported
```
