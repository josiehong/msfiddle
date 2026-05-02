Usage
=====

Installation
------------

.. code-block:: bash

   pip install msfiddle

PyTorch must be installed separately following the
`official PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.
Alternatively, install the optional inference extra:

.. code-block:: bash

   pip install "msfiddle[inference]"

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

Python API
----------

For a single native MS/MS spectrum:

.. code-block:: python

   from msfiddle import predict_from_spectrum

   candidates = predict_from_spectrum(
       mz_array=[60.0, 85.0, 100.0, 125.0, 150.0],
       intensity_array=[10.0, 50.0, 20.0, 35.0, 15.0],
       precursor_mz=180.063,
       adduct="[M+H]+",
       top_k=5,
       instrument_type="orbitrap",
       collision_energy="Unknown",
       device="cpu",
   )

For repeated or batched use, instantiate a predictor once so model checkpoints
are loaded once and reused:

.. code-block:: python

   from msfiddle import MsFiddlePredictor

   predictor = MsFiddlePredictor(instrument_type="orbitrap", device="cpu")
   results = predictor.predict_batch(
       [
           {
               "id": "sample-1",
               "mz_array": [60.0, 85.0, 100.0, 125.0, 150.0],
               "intensity_array": [10.0, 50.0, 20.0, 35.0, 15.0],
               "precursor_mz": 180.063,
               "adduct": "[M+H]+",
               "collision_energy": "Unknown",
           }
       ]
   )

MGF files can also be used from Python:

.. code-block:: python

   from msfiddle import predict_from_mgf

   df = predict_from_mgf(
       "/path/to/data.mgf",
       instrument_type="orbitrap",
       device="cpu",
   )

The Python APIs are quiet by default and do not download checkpoints unless
``download_models=True`` is passed. The CLI keeps its existing automatic model
download behavior.
