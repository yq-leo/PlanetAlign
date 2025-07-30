Benchmarking Scalability
=========================

This tutorial shows how to benchmark the **scalability** of a network alignment (NA) algorithm in ``PlanetAlign``, which is a also critical factor when evaluating network alignment (NA) algorithms, especially on large graphs.

We demonstrate how to benchmark:

1. **Runtime**: the training and inference time of the alignment algorithm.
2. **Memory usage**: the peak RAM or GPU memory consumed during alignment.

Both benchmarks can be used to study the trade-off between accuracy and efficiency, and to compare methods under realistic deployment constraints.

.. contents::
   :local:
   :depth: 2

Measuring Runtime
-------------------------

To evaluate alignment efficiency of a NA algorithm on a specific dataset, you can access the training time from the ``TrainLogger`` object and access the inference time from the
return dictionary of ``.test()`` method.

Here's an example using the ``JOENA`` algorithm on the ``PhoneEmail`` dataset:

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

    # Evaluate using built-in metrics (training time)
    result = model.test(data, gids=[0, 1], metrics=['time'])
    
    # Print the training time and inference time
    print(f"Training time: {logger.log_metric('time'):.4f} seconds")
    print(f"Inference time: {result['time']:.4f} seconds")

.. note::

    For more accurate timing, consider repeating multiple runs and averaging results.

Measuring Memory Usage
-----------------------

Similarily, to evaluate alignment efficiency of a NA algorithm on a specific dataset, you can access the peak memory usage during training and testing from 
the ``TrainLogger`` object and the return dictionary of ``.test()`` method, respectively.

Here is an example of how to benchmark **peak memory usage** of a NA algorithm.

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

    # Evaluate using built-in metrics (peak memory usage)
    result = model.test(data, gids=[0, 1], metrics=['mem'])
    
    # Print the peak memory usage during training and inference
    print(f"Peak memory during training: {logger.log_metric('mem'):.4f} GB")
    print(f"Peak memory during inference: {result['mem']:.4f} GB")

Best Practices
--------------

- Always separate **training** and **inference** time to avoid confounding results.
- Benchmark multiple seeds or dataset sizes to understand scalability trends.
- Plot time/memory vs. graph size if conducting large-scale analysis.

Summary
-------

In this tutorial, we demonstrated how to measure the **runtime** and **memory usage** of network alignment algorithms using PlanetAlign’s built-in tools.

- You can benchmark **training time** using the ``TrainLogger``, and **inference time** via the return dictionary of ``.test()`` with `metrics=['time']`.
- Similarly, you can track **peak memory usage** during both training and inference using ```metrics=['mem']``.
- These tools allow you to quantify the **computational cost** of different NA algorithms, enabling fair comparisons beyond accuracy alone.

Next: See the "Robustness Analysis" section to understand how to benchmark the performance of NA algorithms under noisy conditions.

