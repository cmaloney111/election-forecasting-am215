Development Guide
=================

Setting Up Development Environment
-----------------------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/yourusername/election-forecasting.git
   cd election-forecasting
   uv pip install -e .
   uv sync

Running Tests
-------------

Run the test suite:

.. code-block:: bash

   make test

Run tests with coverage:

.. code-block:: bash

   make test-cov

This generates:

* Terminal coverage report
* HTML coverage report in ``htmlcov/``
* XML coverage report for CI tools

Code Quality
------------

Linting
~~~~~~~

Check code style and formatting:

.. code-block:: bash

   make lint

Auto-format code:

.. code-block:: bash

   make format

Quality Check
~~~~~~~~~~~~~

Run both linting and tests:

.. code-block:: bash

   make quality-check

Building Documentation
----------------------

Build the Sphinx documentation:

.. code-block:: bash

   make docs

The documentation will be built to ``docs/build/html/``. Open ``docs/build/html/index.html`` in a browser to view.

Building Distribution Packages
-------------------------------

Build source and wheel distributions:

.. code-block:: bash

   make build

This creates:

* ``dist/election_forecasting-X.Y.Z.tar.gz`` (source distribution)
* ``dist/election_forecasting-X.Y.Z-py3-none-any.whl`` (wheel)

Profiling
~~~~~~~~~

Profile model performance using the built-in ``--profile`` flag:

.. code-block:: bash

   # Profile a quick run with 2 forecast dates
   election-forecast --profile forecast.prof --dates 2

   # View the profile with snakeviz
   snakeviz forecast.prof

Or use the Makefile shortcut:

.. code-block:: bash

   make profile        # Runs forecast with profiling
   make profile-view   # Opens snakeviz to view results