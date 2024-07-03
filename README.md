# integrated_plant_protection
[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/integrated-plant-protection/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/integrated-plant-protection/job/main/)

Integrated Plant Protection

To launch it, first install the package then run [deepaas](https://github.com/ai4os/DEEPaaS):
```bash
git clone https://github.com/ai4eosc-psnc/integrated_plant_protection
cd integrated_plant_protection
pip install -e .
deepaas-run --listen-ip 0.0.0.0
```
The associated Docker container for this module can be found in https://github.com/ai4eosc-psnc/DEEP-OC-integrated_plant_protection.

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
