========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |appveyor|
        | |coveralls| |codecov|
        | |codacy| |codeclimate|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/irregular-object-packing/badge/?style=flat
    :target: https://irregular-object-packing.readthedocs.io/
    :alt: Documentation Status

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/MbBrainz/irregular-object-packing?branch=main&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/MbBrainz/irregular-object-packing

.. |github-actions| image:: https://github.com/MbBrainz/irregular-object-packing/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/MbBrainz/irregular-object-packing/actions

.. |coveralls| image:: https://coveralls.io/repos/MbBrainz/irregular-object-packing/badge.svg?branch=main&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/MbBrainz/irregular-object-packing

.. |codecov| image:: https://codecov.io/gh/MbBrainz/irregular-object-packing/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/MbBrainz/irregular-object-packing

.. |codacy| image:: https://img.shields.io/codacy/grade/MbBrainz/irregular-object-packing.svg
    :target: https://www.codacy.com/gh/MbBrainz/irregular-object-packing
    :alt: Codacy Code Quality Status

.. .. |codeclimate| image:: https://codeclimate.com/github/MbBrainz/irregular-object-packing/badges/gpa.svg
..    :target: https://codeclimate.com/github/MbBrainz/irregular-object-packing
..    :alt: CodeClimate Quality Status

.. |version| image:: https://img.shields.io/pypi/v/irregular-object-packing.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/irregular-object-packing

.. |wheel| image:: https://img.shields.io/pypi/wheel/irregular-object-packing.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/irregular-object-packing

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/irregular-object-packing.svg
    :alt: Supported versions
    :target: https://pypi.org/project/irregular-object-packing

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/irregular-object-packing.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/irregular-object-packing

.. |commits-since| image:: https://img.shields.io/github/commits-since/MbBrainz/irregular-object-packing/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/MbBrainz/irregular-object-packing/compare/v0.0.0...main



.. end-badges

A high performance 3D oject packing library for irregularly shaped objects.

* Free software: MIT license

Installation
============

::

    pip install irregular-object-packing

You can also install the in-development version with::

    pip install https://github.com/MbBrainz/irregular-object-packing/archive/main.zip


Documentation
=============


https://irregular-object-packing.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
