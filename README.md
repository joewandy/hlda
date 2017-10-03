# hlda
Gibbs sampler for the Hierarchical Latent Dirichlet Allocation topic model. This is based on the hLDA implementation from Mallet, having a fixed depth on the nCRP tree.

Files
------

- hlda/sampler.py is the Gibbs sampler.
- Example notebook to test with synthetic data and the BBC Insight corpus can be found in the notebooks folder.

References
-----------

- [The Nested Chinese Restaurant Process and Bayesian Nonparametric Inference of Topic Hierarchies](http://cocosci.berkeley.edu/tom/papers/ncrp.pdf)
- [Hierarchical Topic Models and the Nested Chinese Restaurant Process](http://www.cs.columbia.edu/~blei/papers/BleiGriffithsJordanTenenbaum2003.pdf)