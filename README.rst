GIGA-Lens
========================

.. image:: https://img.shields.io/pypi/v/gigalens.svg
    :target: https://pypi.python.org/pypi/gigalens
    :alt: Latest PyPI version

Gradient Informed, GPU Accelerated Lens modelling (GIGA-Lens) is a package for fast Bayesian inference on strong
gravitational lenses. For details, please see `our paper <https://arxiv.org/abs/2202.07663>`__. See
`here <https://giga-lens.github.io/gigalens/>`__ for our documentation.

Usage
-----

Installation
------------
``GIGA-Lens`` can be installed via pip: ::

    pip install gigalens

If pip notes an error after installation about conflicting dependencies, these can usually be safely ignored.
If you wish to test the installation, tests can be run simply by running ``tox`` in the root directory.

If you don’t have access to institutional GPUs, one easy way is to use GPU on Google Colab.  Please remember the
very first cell should have ``!pip install gigalens``. If you do have access to institutional GPUs, you can set up a
notebook to run on GPU.  For example, at `NESRC <https://jupyter.nersc.gov/hub/>`__, you can choose the kernel
``tensorflow-2.6.0``, and include in the first cell: ``!pip install gigalens``.


Requirements
^^^^^^^^^^^^
The following packages are requirements for GIGA-Lens. However, ``!pip install gigalens`` is all you need to do. In fact,
separately installing other packages can cause issues with subpackage dependencies. Some users may find it necessary
to install PyYAML.

::

    tensorflow>=2.6.0
    tensorflow-probability>=0.15.0
    lenstronomy==1.9.3
    scikit-image==0.18.2
    tqdm==4.62.0

The following dependencies are required by ``lenstronomy``:

::

    cosmohammer==0.6.1
    schwimmbad==0.3.2
    dynesty==1.1
    corner==2.2.1
    mpmath==1.2.1



Authors
-------

`GIGALens` was written by `Andi Gu <andi.gu@berkeley.edu>`_.
