File Formats
============

Input: MGF
----------

msfiddle accepts tandem mass spectra in ``.mgf`` format. Four fields are
required per spectrum; all others are ignored.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``TITLE``
     - Unique spectrum identifier, propagated to the ``ID`` column in output.
   * - ``PRECURSOR_MZ``
     - Observed precursor m/z.
   * - ``PRECURSOR_TYPE``
     - Adduct type (e.g. ``[M+H]+``, ``[M-H]-``). See supported types below.
   * - ``COLLISION_ENERGY``
     - Collision energy in eV.

**Example:**

.. code-block:: text

   BEGIN IONS
   TITLE=EMBL_MCF_2_0_HRMS_Library000529
   PEPMASS=111.02016
   CHARGE=1-
   PRECURSOR_TYPE=[M-H]-
   PRECURSOR_MZ=111.02016
   COLLISION_ENERGY=50.0
   41.0148 0.329893
   68.0258 0.402906
   111.0203 100.0
   END IONS

**Supported precursor types:**

``[M+H]+``, ``[M+2H]2+``, ``[M+Na]+``, ``[M-H]-``, ``[M+H-H2O]+``,
``[M-H2O+H]+``, ``[2M+H]+``, ``[2M-H]-``, ``[M+H-2H2O]+``, ``[M+H-NH3]+``,
``[M+H+NH3]+``, ``[M+NH4]+``, ``[M+H-CH2O2]+``, ``[M+H-CH4O2]+``,
``[M-H-CO2]-``, ``[M-CHO2]-``, ``[M-H-H2O]-``

Output: msfiddle CSV
--------------------

One row is produced per input spectrum.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Description
   * - ``ID``
     - Spectrum identifier from the MGF ``TITLE`` field.
   * - ``Y Pred``
     - Raw TCN output: semicolon-separated atom-count vector
       (C, H, O, N, F, S, Cl, P, B, Br, I, Na, K).
   * - ``Mass``
     - Neutral monoisotopic mass derived from ``PRECURSOR_MZ`` and
       ``PRECURSOR_TYPE``.
   * - ``Pred Formula``
     - Top formula from the TCN model prior to refinement.
   * - ``Pred Mass``
     - Monoisotopic mass of ``Pred Formula``.
   * - ``Pred Atom Num``
     - Total atom count predicted by the model.
   * - ``Pred H/C Num``
     - H/C ratio predicted by the model.
   * - ``Running Time``
     - Wall time per spectrum in seconds (prediction + refinement).
   * - ``Refined Formula (k)``
     - The k-th best refined formula (0-indexed), ranked by rescore score.
       ``None`` if fewer than k+1 candidates were found.
   * - ``Refined Mass (k)``
     - Monoisotopic mass of ``Refined Formula (k)``.
   * - ``Rescore (k)``
     - Rescore model confidence score for ``Refined Formula (k)``
       (0–1; higher is more confident).

The number of ranked columns is set by ``top_k`` in the configuration file
(default: 5).

Input: BUDDY CSV (optional)
----------------------------

Required columns when using ``--buddy_path``:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Description
   * - ``ID``
     - Spectrum identifier matching the MGF ``TITLE`` field.
   * - ``Adduct``
     - Precursor type string.
   * - ``Pred Formula (1–5)``
     - Top-5 candidate formulas from BUDDY.
   * - ``BUDDY Score (1–5)``
     - Confidence scores; candidates below the configured threshold are excluded.

Input: SIRIUS CSV (optional)
-----------------------------

Required columns when using ``--sirius_path``:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Description
   * - ``ID``
     - Spectrum identifier matching the MGF ``TITLE`` field.
   * - ``Pred Formula (1–5)``
     - Top-5 candidate formulas from SIRIUS.
   * - ``Pred Adduct (1–5)``
     - Predicted adduct for each candidate.
   * - ``SIRIUS Score (1–5)``
     - Log-likelihood scores; candidates below the configured threshold
       are excluded.
