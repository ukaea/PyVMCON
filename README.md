# PyVMCON
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Testing Workflow](https://github.com/ukaea/PyVMCON/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ukaea/PyVMCON/actions/workflows/tests.yml)

A Python implementation of the VMCON nonlinear constrained optimiser.

## Installation
PyVMCON can be installed from PyPI via:

```
pip install PyVMCON
```

Or installed from source:

```
git clone https://github.com/ukaea/PyVMCON.git
cd PyVMCON
pip install .
```

## Testing
Tests can be run using `pytest tests/`. The tests check the paper examples are replicated. In some cases (I believe due to the way the quadratic programming implementation differs) the values achieved are different to the paper, but still correct; such cases have been noted in the test file.

## Documentation
Documentation for the VMCON algorithm and PyVMCON API/use can be found on our [**GitHub pages**](https://ukaea.github.io/PyVMCON/). The documentation includes references to the VMCON paper and other helpful resources.

## License
PyVMCON is provided under the MIT license, please see the LICENSE file for full details.

Copyright (c) 2023-2024 UK Atomic Energy Authority
