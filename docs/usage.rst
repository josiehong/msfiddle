Usage
=====

Installation
------------

.. code-block:: bash

   pip install msfiddle

PyTorch must be installed separately following the
`official PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

Downloading Pre-trained Models
-------------------------------

Model weights must be downloaded before running predictions:

.. code-block:: bash

   # Download to the default location (~/.msfiddle/check_point)
   msfiddle-download-models

   # Download specific models to a custom location
   msfiddle-download-models --destination /path/to/models \
                             --models fiddle_tcn_qtof fiddle_rescore_qtof

To inspect current model paths:

.. code-block:: bash

   msfiddle-checkpoint-paths

Running Predictions
--------------------

**Demo data:**

.. code-block:: bash

   msfiddle --demo --result_path ./output_demo.csv --device 0

**Custom data:**

.. code-block:: bash

   msfiddle --test_data /path/to/data.mgf \
            --instrument_type orbitrap \
            --result_path /path/to/results.csv \
            --device 0

``--instrument_type`` accepts ``orbitrap`` (default) or ``qtof``.

**Custom model paths:**

.. code-block:: bash

   msfiddle --test_data /path/to/data.mgf \
            --config_path /path/to/config.yml \
            --resume_path /path/to/tcn_model.pt \
            --rescore_resume_path /path/to/rescore_model.pt \
            --result_path /path/to/results.csv \
            --device 0

Integration with BUDDY and SIRIUS
-----------------------------------

Candidate formulas from `BUDDY <https://github.com/Philipp-Sc/buddy>`_ and
`SIRIUS <https://bio.informatik.uni-jena.de/software/sirius/>`_ can be
incorporated to improve refinement results:

.. code-block:: bash

   msfiddle --test_data /path/to/data.mgf \
            --buddy_path /path/to/buddy_results.csv \
            --sirius_path /path/to/sirius_results.csv \
            --result_path /path/to/results.csv \
            --device 0

See :doc:`formats` for the required CSV format for BUDDY and SIRIUS inputs.
