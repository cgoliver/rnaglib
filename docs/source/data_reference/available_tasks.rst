Available Benchmarking Tasks 
==============================

The new tasks module allows the use and creation of a variety of machine learning tasks on RNA structure. The list and description of the task is found below, followed by tutorials on using existing tasks, as well as on developing new tasks.

You can load any task using its Task ID as such::

    >>> from rnaglib.tasks import get_task
    >>> task = get_task(root="myroot", task_id="RNA_CM")

To get a list of all available task IDs::

    >>> from rnaglib.tasks import TASKS
    >>> TASKS
    ["RNA_CM",
     "RNA_IF",
     ...
     ]


.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Task ID 
     - Description
     - Class
     - Source
   * - ``RNA_CM``
     - Prediction of chemical modifications at the residue level.
     - ``ChemicalModification``
     - No published instance
   * - ``RNA_IF``
     - Prediction of nucleotide identity (sequence) at each residue.
     - ``InverseFolding``
     - 
   * - 
     - Prediction of nucleotide identiy at each residue using data and splits from ``gRNAde``.
     - ``gRNAde``
     - [Joshi_et_al_2024]_
   * - ``RNA_VS``
     - Scoring of candidates in virtual screening scenario based on ``RNAmigos 2.0``.
     - ``VSTask``
     - [Carvajal-Patino_2023]_
   * - ``RNA_Site``
     - Prediction of whether a residue is part of a binding site.
     - ``BindingSiteDetection``
     - 
   * - 
     - Prediction of whether a residue is part of a binding site using data and splits from ``RNASite``
     - ``BenchmarkLigandBindingSiteDetection``
     - [Su_et_al_2021]_
   * - ``RNA_Ligand``
     - Prediction of ligand identity given a binding pocket (RNA structure subgraph) using data and splits from ``GMSM``.
     - ``GMSM``
     - [Pellizzoni_et_al_2024]_
   * - ``RNA_Prot``
     - Prediction of whether a residue is part of a protein binding site.
     - ``ProteinBindingSiteDetection``
     - [Wang_et_al_2018]_

We provide a short tutorial on (1) using an existing tasks to assess model perfomance and (2) building custom tasks using modular `rnaglib` functionality.

Code to reproduce the results included in the correspoding submission can be found in the `experiments/` directory.


.. [Carvajal-Patino_2023] Semi-supervised learning and large-scale docking data accelerate rna virtual screening. bioRxiv, pages 2023–11, 2023.

.. [Joshi_et_al_2024] Chaitanya K Joshi, Arian R Jamasb, Ramon Viñas, Charles Harris, Simon V Mathis, Alex Morehead, and Pietro Liò. gRNAde: Geometric deep learning for 3d rna inverse design. bioRxiv, pages 2024–03, 2024.

.. [Pellizzoni_et_al_2024] Paolo Pellizzoni, Carlos Oliver, and Karsten Borgwardt. Structure- and function-aware substitution matrices via learnable graph matching. In Research in Computational Molecular Biology, 2024.

.. [Su_et_al_2021] Hong Su, Zhenling Peng, and Jianyi Yang. Recognition of small molecule–rna binding sites using RNA sequence and structure. Bioinformatics, 37(1):36–42, 2021.

.. [Wang_et_al_2018] Kaili Wang, Yiren Jian, Huiwen Wang, Chen Zeng, and Yunjie Zhao. Rbind: computational network method to predict rna binding sites. Bioinformatics, 34(18):3131–3136, 2018.


