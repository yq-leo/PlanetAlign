Working with NA Algorithms
===========================

``PlanetAlign`` provides a standardized interface for evaluating, training, and comparing network alignment (NA) algorithms. It includes a collection of **built-in alignment models** and a **base class** :class:`BaseModel` that can be extended to define custom methods.

This tutorial covers:

1. How to run built-in NA algorithms (e.g., ``FINAL``)
2. How to define your own alignment model by subclassing :class:`BaseModel`

Whether you're benchmarking performance or building new models, this section will guide you through the key workflows.

.. contents::
   :local:
   :depth: 2

Using Built-in Algorithms
--------------------------

PlanetAlign includes a collection of built-in network alignment algorithms such as ``FINAL``, ``PARROT``, and ``JOENA``. These models are accessible under the ``PlanetAlign.algorithms`` module and follow a consistent interface for training and testing.

Here’s an example of running the ``FINAL`` algorithm on the ``Douban`` dataset:

.. code-block:: python

    from PlanetAlign.datasets import Douban
    from PlanetAlign.algorithms import FINAL

    # Load the dataset
    data = Douban()

    # Initialize the algorithm
    model = FINAL()

    # Train the model on source and target graphs
    model.train(data,  gids=[0, 1])

    # Test and evaluate the model
    result = model.test(data, gids=[0, 1])

    print("Evaluation Results:", result)

.. note::

    All built-in models work out-of-the-box with datasets derived from :class:`BaseData` and :class:`Dataset` classes of PLANETALIGN.

Customizing NA Algorithms
--------------------------

To implement your own network alignment model, you can subclass the :class:`PlanetAlign.algorithms.BaseModel` class. This ensures compatibility with the training, prediction, and evaluation pipeline used across ``PlanetAlign``.

Here's a minimal example of creating a custom algorithm that randomly matches nodes:

.. code-block:: python

    import random
    from PlanetAlign.algorithms import BaseModel

    class RandomAligner(BaseModel):
        def train(self, data, gids):
            g1, g2 = data.pyg_graphs[gids[0]], data.pyg_graphs[gids[1]]
            S = torch.rand(n1, n2)
            self.S = torch.nn.functional.softmax(self.S, dim=1)
            return self.S

    # Use the custom model
    from PlanetAlign.datasets import Douban
    data = Douban()

    model = RandomAligner()
    model.train(data)
    result = model.test(data, pred_alignment)
    print("Random baseline result:", result)

.. note::

    To ensure reproducibility and compatibility:

    - Your model must implement ``train(self, data)``
    - ``test(self, data)`` is inherited from :class:`BaseModel` but can be overridden if needed.

Summary
-------

- Use :class:`PlanetAlign.algorithms.Model` to quickly run built-in NA algorithms.
- Create new models by subclassing :class:`BaseModel` and implementing ``train()``.
- PlanetAlign’s standardized pipeline ensures consistent training and evaluation across all algorithms.
