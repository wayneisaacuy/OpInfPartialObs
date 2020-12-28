# OpInfPartialObs

Operator inference for partial observations to accompany the manuscript ***Operator inference for learning reduced models with non-Markovian terms from partially observed state trajectories*** by W.I.T. Uy and B. Peherstorfer. The code and data reproduce the numerical examples in the appendix in which we numerically compare the error between the Markovian reduced model and the reduced model with non-Markovian term.

The main scripts are <code>LowPartialObsLowDim.py</code> (low rate of observed state components and small reduced dimension) and <code>HighPartialObsHighDim.py</code> (high rate of observed state components and large reduced dimension). The helper functions are contained in <code>utils.py</code>. These reproduce Figure 17 (left and right panel) of the manuscript.

