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

Publishing to PyPI
------------------

Test PyPI (Recommended First)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create account on https://test.pypi.org
2. Generate API token
3. Configure ``~/.pypirc`` with token (see ``.pypirc.template``)
4. Upload to Test PyPI:

.. code-block:: bash

   make build
   make upload-test

5. Test installation:

.. code-block:: bash

   pip install --index-url https://test.pypi.org/simple/ election-forecasting

Production PyPI
~~~~~~~~~~~~~~~

1. Create account on https://pypi.org
2. Generate API token
3. Configure ``~/.pypirc`` with token
4. Upload to PyPI:

.. code-block:: bash

   make build
   make upload

Project Structure
-----------------

::

   election-forecasting/
   ├── election_forecasting/         # Main package
   │   ├── models/                   # Forecasting models
   │   │   ├── base_model.py        # Abstract base class
   │   │   ├── poll_average.py      # Simple baseline
   │   │   ├── kalman_diffusion.py  # Kalman filter model
   │   │   ├── improved_kalman.py   # Enhanced Kalman
   │   │   └── hierarchical_bayes.py # Best model
   │   ├── scripts/                  # CLI entry points
   │   │   ├── run_all_models.py    # Run all models
   │   │   ├── compare_models.py     # Compare performance
   │   │   ├── generate_plots.py     # Create visualizations
   │   │   └── run_all.py           # Complete pipeline
   │   └── utils/                    # Utility functions
   │       └── data_utils.py         # Data loading & metrics
   ├── tests/                        # Test suite
   │   ├── conftest.py              # Pytest fixtures
   │   ├── test_base_model.py       # Base class tests
   │   ├── test_data_utils.py       # Data utility tests
   │   └── test_models.py           # Model tests
   ├── docs/                         # Documentation
   │   └── source/                   # Sphinx source files
   ├── data/                         # Input data
   │   ├── polls/                    # FiveThirtyEight polls
   │   └── election_results/         # MIT Election Lab results
   ├── predictions/                  # Model output (CSV)
   ├── metrics/                      # Evaluation metrics (TXT)
   ├── plots/                        # Visualizations (PNG)
   ├── pyproject.toml               # Package configuration
   ├── Makefile                      # Development commands
   └── README.md                     # Project overview

Contributing
------------

Branching Strategy
~~~~~~~~~~~~~~~~~~

Use feature branches:

.. code-block:: bash

   git checkout -b feature/my-feature
   # Make changes, commit
   git checkout main
   git merge feature/my-feature

Commit Messages
~~~~~~~~~~~~~~~

Use semantic commit prefixes:

* ``feat:`` New features
* ``fix:`` Bug fixes
* ``docs:`` Documentation changes
* ``test:`` Test additions/changes
* ``refactor:`` Code refactoring
* ``perf:`` Performance improvements
* ``build:`` Build system changes
* ``ci:`` CI/CD changes
* ``chore:`` Maintenance tasks

Pull Requests
~~~~~~~~~~~~~

1. Ensure all tests pass: ``make quality-check``
2. Add tests for new functionality
3. Update documentation as needed
4. Write clear commit messages
5. Submit PR with description of changes

Debugging
---------

Verbose Mode
~~~~~~~~~~~~

Run commands with ``-v`` for detailed output:

.. code-block:: bash

   election-forecast -v

Profiling
~~~~~~~~~

Profile model performance:

.. code-block:: bash

   python -m cProfile -o output.prof election_forecasting/scripts/run_all_models.py

Analyze with snakeviz:

.. code-block:: bash

   snakeviz output.prof

Interactive Debugging
~~~~~~~~~~~~~~~~~~~~~

Use IPython for interactive exploration:

.. code-block:: python

   from election_forecasting.models.hierarchical_bayes import HierarchicalBayesModel
   model = HierarchicalBayesModel()

   # Load data
   polls, results = model.load_data()

   # Inspect specific state
   fl_polls = polls[polls['state_code'] == 'FL']
   print(fl_polls.head())
