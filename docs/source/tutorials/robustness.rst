Benchmarking Robustness
========================

Robustness is also a critical aspect of evaluating network alignment (NA) algorithms, especially when working with **noisy real-world data**.
This tutorial shows how to benchmark the **robustness** of a network alignment (NA) algorithm in ``PlanetAlign`` against various types of noise that can occur in practice, such as:

- Edge noise (randomly added or removed edges)
- Attribute noise (corrupted node features)
- Supervision noise (incorrect anchor links)

In this tutorial, we test the **robustness** of NA algorithms by injecting controlled noise into:

1. Graph structure (edge noise)
2. Node features (attribute noise)
3. Supervision (supervision noise)

``PlanetAlign`` provides utility functions in ``PlanetAlign.utils`` compatible with the built-in datasets and custom datasets derived from :class:`BaseData`
to facilitate such perturbations and evaluate model performance under varying levels of noise.

.. contents::
   :local:
   :depth: 2

Adding Edge Noise
-----------------

To simulate structural perturbation, you can add random edges to one or both graphs using

.. autofunction:: PlanetAlign.utils.add_edge_noises

Here is an example that perturbs ``G1`` and ``G2`` with 10% edge noise:

.. code-block:: python

    from PlanetAlign.datasets import Douban
    from PlanetAlign.utils import add_edge_noises

    data = Douban()
    
    # Add edge noise to both graphs with noise_rate = 0.1
    data = add_edge_noises(data, noise_rate=0.1, gids=[0, 1], inplace=False)

    # Check the new number of edges
    print("G1 edges (after noise):", data.pyg_graphs[0].num_edges)
    print("G2 edges (after noise):", data.pyg_graphs[1].num_edges)

.. note::

    - ``noise_rate`` ∈ [0, 1] determines the fraction of new random edges added.
    - ``gids`` specifies which graphs (by index) to perturb (e.g., `[0]`, `[1]`, or `[0, 1]`)
    - ``inplace`` argument is defaulted to ``False``, meaning it returns a new dataset object with noise applied. Set ``inplace=True`` to modify the original dataset directly.

Adding Attribute Noise
-----------------------

To test the model's robustness to noisy node features, you can use:

.. autofunction:: PlanetAlign.utils.add_attr_noises

This randomly corrupts a proportion of feature vectors by replacing them with noise (e.g., Gaussian or uniform).

.. code-block:: python

    from PlanetAlign.datasets import Douban
    from PlanetAlign.utils import add_attr_noises

    data = Douban()

    # Add 20% attribute noise to both graphs by fliping binary attributes
    data = add_attr_noises(data, mode='flip', noise_rate=0.2, gids=[0, 1], inplace=False)

    # Check that node features are still the same shape
    print("X1 shape:", data.X1.shape)
    print("X2 shape:", data.X2.shape)

.. note::

    - Only applies if ``X1`` and/or ``X2`` are present.
    - For binary features, use ``mode='flip'`` to randomly flip bits.
    - For continuous features, use ``mode='gaussian'`` to add Gaussian noise.

Adding Supervision Noise
-------------------------

To simulate incorrect alignment supervision (e.g., noisy anchors), use:

.. code-block:: python

    # Add supervision noise to graphs in a PyNetAlign dataset by injecting noisy anchors.
    add_sup_noises(dataset, noise_ratio, src_gid=0, dst_gid=1, seed=None, inplace=False)

**Parameters**

- **dataset** (*Dataset*) – The input dataset containing the graphs and ground-truth alignment.
- **noise_ratio** (*float*) – The ratio of supervision to perturb (value between 0 and 1).
- **src_gid** (*int*, *optional*) – The graph ID of the source graph. Default is ``0``.
- **dst_gid** (*int*, *optional*) – The graph ID of the destination graph. Default is ``1``.
- **seed** (*int*, *optional*) – Random seed for reproducibility.
- **inplace** (*bool*, *optional*) – If ``True``, modify the dataset in place. Otherwise, return a new dataset with supervision noise applied.

**Returns**

- **Dataset** – A PyG dataset with perturbed supervision (modified anchors).

**Return type**

- :class:`Dataset`

This randomly replaces a proportion of training anchor pairs with mismatched ones:

.. code-block:: python

    from PlanetAlign.datasets import Douban
    from PlanetAlign.utils import add_sup_noises

    data = Douban()

    # Add 30% supervision noise into anchor links for training
    data = add_sup_noises(data, noise_rate=0.3, inplace=False)

.. note::

    - ``noise_rate`` controls the fraction of anchor pairs corrupted.
    - This helps test how models perform when supervision is imperfect or partially mislabeled.

Best Practices
--------------

- Run multiple trials per noise level to reduce variance in evaluation.
- Plot metric degradation (e.g., MRR, Hits@K) versus noise rate to analyze robustness curves.
- Vary only one noise source at a time (e.g., edge vs. attribute) to isolate effects.

Summary
-------

In this tutorial, we showed how to:

- Inject **edge noise** using :func:`add_edge_noises`
- Add **attribute noise** using :func:`add_attr_noises`
- Simulate **supervision noise** using :func:`add_sup_noises`

These utilities allow you to evaluate how sensitive NA algorithms are to real-world imperfections. Next, consider benchmarking robustness across multiple datasets or noise regimes for more comprehensive analysis.
