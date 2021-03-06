# OpInfPartialObs

Operator inference for partial observations to accompany the manuscript [Operator inference of non-Markovian terms for learning reduced models from partially observed state trajectories (arXiv:2103.01362)](https://arxiv.org/abs/2103.01362) by W.I.T. Uy and B. Peherstorfer. The code and data reproduce the numerical examples in the appendix in which we numerically compare the error between the Markovian reduced model and the reduced model with non-Markovian term.

The main scripts are <code>LowPartialObsLowDim.py</code> (low rate of observed state components and small reduced dimension) and <code>HighPartialObsHighDim.py</code> (high rate of observed state components and large reduced dimension). The helper functions are contained in <code>utils.py</code>. These reproduce Figure 17 (left and right panel) of the manuscript.

<pre><code>@ARTICLE{UyP2021,
       author = {{Uy}, Wayne Isaac Tan and {Peherstorfer}, Benjamin},
        title = "{Operator inference of non-Markovian terms for learning reduced models from partially observed state trajectories}",
      journal = {arXiv e-prints},
     keywords = {Mathematics - Numerical Analysis, Computer Science - Machine Learning},
         year = 2021,
        month = march,
          eid = {arXiv:2103.01362},
        pages = {arXiv:2103.01362},
archivePrefix = {arXiv},
       eprint = {2103.01362},
 primaryClass = {math.NA},
}
</code></pre>
