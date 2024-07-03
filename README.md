# integrated_plant_protection
[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/integrated-plant-protection/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/integrated-plant-protection/job/main/)

Integrated Plant Protection

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
├── LICENSE                <- License file
│
├── README.md              <- The top-level README for developers using this project.
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
│                             generated with `pip freeze > requirements.txt`
│
├── setup.py, setup.cfg    <- makes project pip installable (pip install -e .) so
│                             integrated_plant_protection can be imported
│
├── integrated_plant_protection    <- Source code for use in this project.
│   │
│   ├── __init__.py        <- Makes integrated_plant_protection a Python module
│   │
│   └── api.py             <- Main script for the integration with DEEP API
│
└── Jenkinsfile            <- Describes basic Jenkins CI/CD pipeline
```
