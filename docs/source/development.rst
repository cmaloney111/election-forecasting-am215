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

Profile sequential execution:

.. code-block:: bash

   make profile        # Profile with 8 forecast dates (sequential)
   make profile-view   # Opens snakeviz to view results

Profile parallel execution:

.. code-block:: bash

   make profile-parallel        # Profile with 8 dates, 4 workers
   make profile-view-parallel   # Opens snakeviz for parallel profile

The profiling targets help identify performance bottlenecks and compare sequential vs parallel execution overhead.

Parallel Execution
------------------

The forecasting models support parallel execution via the ``--parallel`` flag:

.. code-block:: bash

   # Sequential execution (default)
   election-forecast --dates 16

   # Parallel execution with 4 workers
   election-forecast --dates 16 --parallel 4

**How it works:**

* Parallelization is done at the **forecast date level**
* Each worker processes all states for a single date
* Most beneficial with 8+ forecast dates
* Maintains reproducibility with ``--seed`` argument

**Performance characteristics:**

* Best speedup with many dates (16+) on multi-core machines
* Process spawning overhead can dominate for small workloads
* Recommended: Use 4 workers on typical quad-core machines

Docker Development
------------------

Build and test using Docker:

.. code-block:: bash

   # Build the image
   docker build -t election-forecasting .

   # Run tests in container
   docker run election-forecasting make test

   # Run forecasts with volume mounts
   docker run -v $(pwd)/predictions:/app/predictions \
              -v $(pwd)/metrics:/app/metrics \
              election-forecasting election-forecast --dates 8

   # Use parallel execution in Docker
   docker run -v $(pwd)/predictions:/app/predictions \
              -v $(pwd)/metrics:/app/metrics \
              election-forecasting election-forecast --dates 16 --parallel 4

The Docker container includes all dependencies and ensures reproducible builds across platforms.