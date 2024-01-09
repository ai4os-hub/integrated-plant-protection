# integrated_plant_protection
[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/UC-ai4eosc-psnc-integrated_plant_protection/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/UC-ai4eosc-psnc-integrated_plant_protection/job/master)

Integrated Plant Protection

To launch it, first install the package then run [deepaas](https://github.com/indigo-dc/DEEPaaS):
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
