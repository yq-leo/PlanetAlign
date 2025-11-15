Benchmarking Effectiveness
==========================

This tutorial shows how to benchmark the **effectiveness** of a network alignment (NA) algorithm in ``PlanetAlign`` on a given dataset, which is crucial for evaluating model quality and comparing different methods.

We cover two main topics:

1. Evaluating a model using standard metrics like **Hits@K** and **Mean Reciprocal Rank (MRR)**.
2. Visualizing the **training curve** to track trends over epochs (e.g., Hits@K, MRR).

These tools are essential for understanding model quality, selecting hyperparameters, and comparing competing methods.

.. contents::
   :local:
   :depth: 2

Evaluating Effectiveness
-------------------------

After training a model using ``.train()`` method of the NA algorithm, you can evaluate its alignment quality using ``.test()`` method. By default, this includes:

- **Hits@K**: Measures the proportion of correct alignments within the top-K predictions.
- **MRR (Mean Reciprocal Rank)**: Measures the average reciprocal rank of the correct alignment.

Here is an example that benchmarks the ``FINAL`` algorithm on the ``PhoneEmail`` dataset:

.. code-block:: python

    from PlanetAlign.datasets import PhoneEmail
    from PlanetAlign.algorithms import FINAL
    from PlanetAlign.logger import TrainLogger

    # Load dataset
    data = PhoneEmail(root='datasets/')

    # Initialize and train model, add .to('cuda') if using GPU
    model = FINAL()
    # Initialize logger to track the training process
    logger = TrainLogger(log_path='logs/', save=True)
    # Train the model to align the first and second graphs
    model.train(data, gids=[0, 1], logger=logger)

    # Evaluate using built-in metrics
    result = model.test(data, gids=[0, 1])

    print("Evaluation metrics:")
    for metric, value in result.items():
        print(f"{metric}: {value:.4f}")

.. note::

    By default, ``.test()`` method reports `Hits@1`, `Hits@10`, `Hits@30`, `Hits@50`, and `MRR`. You can customize this using the `metrics` argument:

    .. code-block:: python

        # Evaluate with Hits@1 and MRR only
        result = model.test(data, gids=[0, 1], metrics=['Hits@1', 'MRR'])

Visualizing the Training Curve
------------------------------

To monitor model convergence and training dynamics, ``PlanetAlign`` provides a built-in utility function: :meth:`plot_curve()`.

This function is available through the model's logger and automatically plots key metrics (e.g., loss, MRR, Hits@K) tracked during training.

If your model logs progress via ``logger``, you can visualize the full training history as follows:

.. code-block:: python

    # Visualize the training curve (MRR) of the model and save the plot
    logger.plot_curve(metric='MRR', save_path='mrr_curve.png')

This will generate a line plot showing metric (MRR) over epochs, helping you identify trends like:

- Convergence behavior
- Optimal hyperparameter settings

.. note::

    For visualization to work properly:
    
    - The model must track metrics over epochs using ``logger.log(epoch, metric, value)``.
    - ``plot_curve()`` automatically aggregates and formats the data for you.

Best Practices
--------------

- Use multiple seeds and report mean ± std across runs for robustness.
- Log per-epoch metrics during training for insight into convergence behavior.

Summary
-------

- Use ``test()`` to compute standard NA metrics like Hits@K and MRR.
- Use built-in function of ``logger`` to visualize training progress for analysis.
- A consistent benchmarking protocol helps compare methods fairly and reproducibly.

Next: See the "Benchmarking Scalability" tutorial to evaluate runtime and memory usage of NA algorithms.

