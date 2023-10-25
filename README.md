# HA_Buffer_Stock
Code for the partial replication of the Heterogenuous Agent model with buffer stock saving as set up in 'The distribution of wealth and the marginal propensity to consume' by Carroll, Slacalek, Tokuoka and White (Quantitative Economics, 2017). This was my final project for the graduate class 'Inequality in Macroeconomics' at TU Vienna in Spring 2023.
The code replicates the simpler model without aggregate shocks, the steady state of which is found using value function iteration. Afterwards, a policy experiment is conductd, in which a stimulus check of about 6.4% of steady state GDP is ins spread out evenly across all agents. The aggregate Marginal Propensities to Consume (MPC's) as well as MPC's of the four wealth quartiles that are implied by this experiment are computed. The path of all aggregate variables is projected using the shooting method.

An overview of the model, the policy experiment and the results can be found in the [slides](https://github.com/valwinkler/HA_Buffer_Stock/blob/main/Slides_HA_Buffer_Stock.pdf) in the repository.

The code still has some shortcomings:
* While the steady state values without ex-ante heterogeneity pretty much match the values of the original papers, wealth inequality in my model *with* ex-ante heterogeneity is somehow lower than in the same model of the original paper. I did not yet find out if this is due to an economic modelling issue or a computational problem from my side.
* The aggregate variables after the government stimulus do not converge too smoothly and take a long time to return to the steady state (as can be seen in the slides). 
* Even thought I used just in time compilation with `numba`, the code takes a rather long time to run (~12-24h on a MacBook pro). Most likely this is due to many (72) interpolations being conducted for every value function evaluation. I can imagine that it would be possible to speed up the code.

If you have any suggestions regarding these three points or have general feedback, [email me](mailto:valentin.winkler@icloud.com).
