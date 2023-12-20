# ai4eosc_uc2
[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/UC-ai4eosc-psnc-ai4eosc_uc2/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/UC-ai4eosc-psnc-ai4eosc_uc2/job/master)

Integrated Plant Protection

To launch it, first install the package then run [deepaas](https://github.com/indigo-dc/DEEPaaS):
```bash
git clone https://github.com/ai4eosc-psnc/ai4eosc_uc2
cd ai4eosc_uc2
pip install -e .
deepaas-run --listen-ip 0.0.0.0
```
The associated Docker container for this module can be found in https://github.com/ai4eosc-psnc/DEEP-OC-ai4eosc_uc2.

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
│                             ai4eosc_uc2 can be imported
│
├── ai4eosc_uc2    <- Source code for use in this project.
│   │
│   ├── __init__.py        <- Makes ai4eosc_uc2 a Python module
│   │
│   └── api.py             <- Main script for the integration with DEEP API
│
└── Jenkinsfile            <- Describes basic Jenkins CI/CD pipeline
```
