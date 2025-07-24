Quick Start with PLANETALIGN
=============================

This guide provides a minimal working example for running a built-in network alignment (NA) method on a built-in dataset using PLANETALIGN.

.. contents::
   :local:
   :depth: 2

Installation
------------

First, install PLANETALIGN via pip (recommended in a virtual environment):

.. code-block:: bash

    pip install planetalign

You may also install from source:

.. code-block:: bash

    git clone https://github.com/yq-leo/PlanetAlign.git
    cd PlanetAlign
    pip install -e .

Basic Usage
-----------

Here is a minimal example of aligning two social networks (Douban dataset) using the ``FINAL`` algorithm:

.. code-block:: python

    import PlanetAlign

    # Download and load the Douban dataset
    dataset = PlanetAlign.datasets.Douban(
        root='data/',
        download=True,
        train_ratio=0.2,
        seed=42
    )

    # Initialize the FINAL alignment model
    model = PlanetAlign.algorithms.FINAL(
        alpha=0.9,  # hyperparameter specific to FINAL
    ).to('cuda')  # or 'cpu'

    # Initialize logger
    logger = PlanetAlign.logger.TrainLogger(
        log_dir='logs/',
        log_name='final_douban',
        save=True
    )

    # Train the model
    model.train(
        dataset=dataset,
        gid1=0,          # index of the first graph
        gid2=1,          # index of the second graph
        use_attr=True,      # use attributes if available
        logger=logger,
        total_epochs=50
    )

    # Evaluate the model
    result = model.test(
        dataset=dataset,
        gids=[0, 1],
        metrics=['Hits@1', 'Hits@10', 'MRR']
    )

    print(result)

Visualizing Training Metrics
----------------------------

After training, you can visualize metrics like training loss or memory usage:

.. code-block:: python

    logger.plot_curve(metric='Hits@1', save_path='plots/hits1.png')

Next Steps
----------

- Explore other datasets: ``FoursquareTwitter``, ``PhoneEmail``, ``ACM_DBLP``, etc.
- Try other algorithms: ``JOENA``, ``PARROT``, ``NeXtAlign``, etc.
- Define your own dataset or model by inheriting from ``Dataset`` or ``BaseModel``.
