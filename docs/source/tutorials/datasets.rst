Working with NA Datasets
=========================

``PlanetAlign`` builts in various network alignment (NA) datasets whose detailed statistics can be found `here <https://planetalign.readthedocs.io/en/latest/modules/datasets.html>`_. It also offers a base class ``BaseData`` for customizing user-defined datasets to facilitate reproducible research and 
algorithm development. This tutorial demonstrates how to work with network alignment datasets in the ``PlanetAlign`` library, including guidelines for:

- Downloading and loading built-in datasets
- Creating custom datasets using the ``BaseData`` class

.. contents::
   :local:
   :depth: 2

Loading Built-in Datasets
--------------------------

PlanetAlign provides a collection of popular datasets that can be easily downloaded and used for evaluation or training. In this example, we are going to use
`Douban <https://planetalign.readthedocs.io/en/latest/generated/PlanetAlign.datasets.Douban.html#PlanetAlign.datasets.Douban>`_ dataset which consists of two social
network representing user interactions on the Douban website. The dataset provides:

- Two graphs with shared user identities,
- Node attributes for each user, and edge attributes for each relationship/interaction,
- A ground-truth alignment between matched users across two networks.

Here, we show how to download and load Douban dataset using the built-in dataset class:

.. code-block:: python

    from PlanetAlign.datasets import Douban

    # Automatically downloads and loads the Douban dataset
    data = Douban(root='datasets/', download=True, train_ratio=0.2)

    # Inspect the graph structures
    G1, G2 = data.pyg_graphs
    print(f"Graph G1: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
    print(f"Graph G2: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")

    # Access training and testing anchor links
    train_anchor_links = data.train_data
    test_anchor_links = data.test_data
    print(f"# training anchor links: {len(train_anchor_links)}")
    print(f"# testing anchor links: {len(test_anchor_links)}")

Full list of built-in datasets can be found at the ``PlanetAlign.datasets`` module.

Creating a Custom Dataset (with PyG Data)
-------------------------

In addition to built-in datasets, ``PlanetAlign`` supports custom datasets through the base class :class:`PlanetAlign.dataset.BaseData`.

If you're using **PyTorch Geometric**, you can easily integrate your own graphs by passing `torch_geometric.data.Data` objects representing each network. This allows seamless use of existing graph loaders or preprocessing pipelines built with PyG.

We demonstrate this with a synthetic example using two small graphs with node features and an identity alignment.

.. code-block:: python

    import torch
    from torch_geometric.data import Data
    from PlanetAlign.dataset import BaseData

    # Create Graph 1 with 4 nodes and edges
    edge_index_1 = torch.tensor([
        [0, 1, 2, 3],
        [1, 0, 3, 2]
    ], dtype=torch.long)

    x1 = torch.eye(4)  # Node features: identity matrix
    G1 = Data(x=x1, edge_index=edge_index_1)

    # Create Graph 2 with same structure
    edge_index_2 = torch.tensor([
        [0, 1, 2, 3],
        [1, 0, 3, 2]
    ], dtype=torch.long)

    x2 = torch.eye(4)
    G2 = Data(x=x2, edge_index=edge_index_2)

    # Define ground truth alignment (anchor links)
    anchor_links = torch.tensor([
        [0, 0],  # Node 0 in G1 aligns with Node 0 in G2
        [1, 1],  # Node 1 in G1 aligns with Node 1 in G2
        [2, 2],  # Node 2 in G1 aligns with Node 2 in G2
        [3, 3]   # Node 3 in G1 aligns with Node 3 in G2
    ], dtype=torch.long)

    # Wrap in a BaseData object
    data = BaseData(graphs=[G1, G2], 
                    anchor_links=anchor_links, 
                    train_ratio=0.2)

    print("Custom dataset initialized:")
    print("G1:", data.pyg_graphs[0])
    print("G2:", data.pyg_graphs[1])
    print("Anchor links for training:", data.train_data)
    print("Anchor links for testing:", data.test_data)

.. note::

    - ``G1`` and ``G2`` must be PyG ``Data`` objects with `edge_index` defined.
    - Node and edge attributes are optional but recommended for alignment tasks.
    - The ``anchor_links`` tensor should contain pairs of aligned node indices, where each row represents a link between nodes in the two graphs.

Summary
-----

- All built-in datasets are downloadable by setting ``download=True`` in the dataset class constructor.
- The ``BaseData`` class provides a flexible way to create custom datasets while maintaining compatibility with the PlanetAlign framework.

.. tip::

    For advanced users, you can subclass ``BaseData`` to add your own loading logic for large-scale or dynamically generated graphs. APIs designed specifically for dynmaic graphs will be included in future releases.

