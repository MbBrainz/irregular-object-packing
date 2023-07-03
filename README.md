# irregular-object-packing

<!-- ![Documentation Status](https://readthedocs.org/projects/irregular-object-packing/badge/?style=flat)(<https://irregular-object-packing.readthedocs.io/>) -->

![Travis CI](https://img.shields.io/travis/MbBrainz/irregular-object-packing.svg)(<https://travis-ci.org/MbBrainz/irregular-object-packing>)

![PyPi version](https://img.shields.io/pypi/v/irregular-object-packing.svg)(<https://pypi.python.org/pypi/irregular-object-packing>)

![GitHub Actions Build Status](https://github.com/MbBrainz/irregular-object-packing/actions/workflows/github-actions.yml/badge.svg)(<https://github.com/MbBrainz/irregular-object-packing/actions>)

<!-- ![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/MbBrainz/irregular-object-packing?branch=main&svg=true)(<https://ci.appveyor.com/project/MbBrainz/irregular-object-packing>) -->

![Codacy Code Quality Status](https://app.codacy.com/project/badge/Grade/498833b3aa9447c0a6147088c5c9fabd)(<https://www.codacy.com/gh/MbBrainz/irregular-object-packing/dashboard?utm_source=github.com&utm_medium=referral&utm_content=MbBrainz/irregular-object-packing&utm_campaign=Badge_Grade>)

![Commits since latest release](https://img.shields.io/github/commits-since/MbBrainz/irregular-object-packing/v0.0.0.svg)(<https://github.com/MbBrainz/irregular-object-packing/compare/v0.0.0...main>)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Python package for packing irregularly shaped objects in an arbitrary 3D container.
The implementation is based on the paper ["Packing irregular Objects in 3D Space via Hybrid Optimization"](http://dx.doi.org/10.1111/CGF.13490) by Ma Y, Chen Z, Hu W, Wang W 2018.

* Free software: 3-clause BSD license
* Documentation: (COMING SOON!) [https://MbBrainz.github.io/irregular-object-packing](https://MbBrainz.github.io/irregular-object-packing)

## Features

* TODO

## Installation

    pip install irregular-object-packing

You can also install the in-development version with:

    pip install https://github.com/MbBrainz/irregular-object-packing/archive/main.zip

## Documentation

[https://irregular-object-packing.readthedocs.io/](https://irregular-object-packing.readthedocs.io/)

## Known Issues

**Notebook disconnecting (date: 21-mar-2023)**:
If you run the Optimization with jupyter notebook, you may experience the kernel disconnecting and then trying to reconnect.
This is a known issue and the best way current workaround is to downgrade your `jupyter_client` version to `7.3.2` and `tornado` to `6.1` (like in de ./requirements-dev.txt file).
The issue is discussed here: [https://discourse.jupyter.org/t/jupyter-notebook-zmq-message-arrived-on-closed-channel-error/17869/2](https://discourse.jupyter.org/t/jupyter-notebook-zmq-message-arrived-on-closed-channel-error/17869/2)

**Module not found: irregular-object-packing (date: 29 may 2023)**
Fix
