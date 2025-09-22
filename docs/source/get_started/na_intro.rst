Introduction to Network Alignment
===================================

**Network alignment** (NA) aims to find correspondences between the nodes of two networks (or graphs). It plays a crucial role in domains such as social network analysis, bioinformatics, and knowledge graph fusion.

.. contents::
   :local:
   :depth: 2

Problem Definition
----------

.. image:: ../_images/na.pdf
  :align: center
  :width: 1000px

Formally, given two input graphs :math:`\mathcal{G}_1 = \{\mathcal{V}_1, \mathbf{A}_1, \mathbf{X}_1, \mathbf{E}_1\}`, :math:`\mathcal{G}_2 = \{\mathcal{V}_2, \mathbf{A}_2, \mathbf{X}_2, \mathbf{E}_2\}` where:

- :math:`\mathcal{V}_1, \mathcal{V}_2` are node sets,
- :math:`\mathbf{A}_1, \mathbf{A}_2` are graph adjacency matrices,
- :math:`\mathbf{X}_1, \mathbf{X}_2` are node attribute matrics,
- :math:`\mathbf{E}_1, \mathbf{E}_2` are edge attribute matrices,

and a set of anchor links :math:`\mathcal{L} = \{(x, y) \mid x \in \mathcal{V}_1, y \in \mathcal{V}_2\}`, the goal of NA tasks is to learn an alignment matrix :math:`\mathbf{S}` such that :math:`\mathbf{S}(x, y)` reflects the alignment likelihood between node :math:`x` and node :math:`y`.

Variants of NA include:

- **Plain NA**: Only graph topology is available (i.e., no :math:`\mathbf{X}_i` or :math:`\mathbf{E}_i`)
- **Attributed NA**: Node and/or edge attributes are available (i.e., :math:`\mathbf{X}_i` and/or :math:`\mathbf{E}_i` are non-empty)
- **Supervised NA**: Anchor links are provided (i.e., :math:`|\mathcal{L}| > 0`)
- **Unsupervised NA**: No anchor links are available (i.e., :math:`|\mathcal{L}| = 0`)

Categories of Network Alignment Methods
-------------

Network alignment methods can be broadly categorized into the following three classes:

1. **Consistency-based Methods**
   These methods align nodes by preserving structural or attribute consistency between neighborhoods.
   *Examples*: IsoRank, FINAL

2. **Embedding-based Methods**
   These approaches learn node embeddings in a shared space and align based on similarity.
   *Examples*: BRIGHT, NeXtAlign, NetTrans

3. **Optimal Transport (OT)-based Methods**
   These methods formulate alignment as a transport problem over node distributions with learnable cost functions.
   *Examples*: PARROT, JOENA, SLOTAlign

Common Applications
-------------------

- 🧑‍💻 **Social network recommendation**: Align user identities across different social platforms for personalized recommendations.
- 📞 **Communication network alignment** : Align communication patterns between different networks to identify similar users or communities.
- ✍️ **Publication network alignment** : Align research papers or authors for disambiguation.
- 🧬 **Biological network alignment** : Identify orthologous proteins or genes between species.
- 📚 **Knowledge graph fusion** : Merge different knowledge bases into a unified one.
- 🏗️ **Infrastructure network alignment** : Align different infrastructure networks (e.g., transportation, utilities) for better resource management.

