# ldp-audit

A Python package for auditing Local Differential Privacy (LDP) protocols.
LDP-Audit is a work in progress, and we expect to release new versions frequently.
The tool and framework accompanies our PETS 2024 paper:
```
@article{arcolezi2023revealing,
  title={Revealing the True Cost of Local Privacy: An Auditing Perspective},
  author={Arcolezi, H{\'e}ber H and Gambs, S{\'e}bastien},
  journal={arXiv preprint arXiv:2309.01597},
  year={2023}
}
```

## Installation

To install the package and its dependencies, you can use the following commands:

```bash
git clone https://github.com/hharcolezi/ldp-audit.git
cd ldp-audit
pip install -r requirements.txt
```

## Reproducing our paper's experiment

```bash
python experiment_1.py
python experiment_2.py
python experiment_3.py
python experiment_4.py
python experiment_5.py
python experiment_6.py
```

## Usage

```python
# Import necessary modules from ldp-audit package
from ldp_audit.base_auditor import LDPAuditor

# Define audit parameters
seed = 42
nb_trials = int(1e6)
alpha = 1e-2
epsilon = 1
k = 2

print('=====Auditing pure LDP protocols=====')
delta = 0.0
pure_ldp_protocols = ['GRR', 'SS', 'SUE', 'OUE', 'THE', 'SHE', 'BLH', 'OLH']
auditor_pure_ldp = LDP_Auditor(nb_trials=nb_trials, alpha=alpha, epsilon=epsilon, delta=delta, k=k, random_state=seed, n_jobs=-1)

for protocol in pure_ldp_protocols:
    eps_emp = auditor_pure_ldp.run_audit(protocol)
    print("{} eps_emp:".format(protocol), eps_emp)

print('\n=====Auditing approximate LDP protocols=====')
delta = 1e-5
approx_ldp_protocols = ['AGRR', 'ASUE', 'AGM', 'GM', 'ABLH', 'AOLH']
auditor_approx_ldp = LDP_Auditor(nb_trials=nb_trials, alpha=alpha, epsilon=epsilon, delta=delta, k=k, random_state=seed, n_jobs=-1)

for protocol in approx_ldp_protocols:
    eps_emp = auditor_approx_ldp.run_audit(protocol)
    print("{} eps_emp:".format(protocol), eps_emp)
```	
	
## Contact
For any question, please contact [Héber H. Arcolezi](https://hharcolezi.github.io/): heber.hwang-arcolezi [at] inria.fr	
	
## License
This project is licensed under the MIT License. See the [LICENSE file](https://github.com/hharcolezi/ldp-audit/blob/main/LICENSE) for details.	
