Installation
============

Requirements
------------

* Python >= 3.9
* NumPy >= 1.24.0
* Pandas >= 2.0.0
* SciPy >= 1.11.0
* Matplotlib >= 3.7.0
* Rich >= 14.2.0

From PyPI
---------

Install the package using pip:

.. code-block:: bash

   pip install election-forecasting

Or with uv (recommended):

.. code-block:: bash

   uv pip install election-forecasting

From Source
-----------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/cmaloney111/election-forecasting-am215.git
   cd election-forecasting
   uv pip install -e .

Development Installation
------------------------

For development, install with test and documentation dependencies:

.. code-block:: bash

   uv pip install -e .
   uv sync --all-extras

Verify Installation
-------------------

Test that the installation worked:

.. code-block:: bash

   election-forecast --help
   python -c "import election_forecasting; print(election_forecasting.__file__)"
