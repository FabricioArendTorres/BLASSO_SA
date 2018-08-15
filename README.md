# BLASSO_SA
MAP estimation of the Bayesian LASSO via Simulated Annealing.

# Sampler
The Simulated Annealing is based on the Gibbs sampler presented in [1] (with marginalized out &#956;).
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{array}{llcl}&space;&&&A&space;=&space;X^TX&space;&plus;&space;D_\tau^{-1}&space;\\&space;&&&D_\tau&space;=&space;diag(\tau_1^2,\dots,\tau_p^2)&space;\\&space;\\&space;\beta&space;&|&space;\quad&space;\tilde{y},&space;\sigma^2,&space;\tau_1^2,&space;\dots,&space;\tau_p^2&space;&\sim&&space;\mathcal{N}(A^{-1}&space;X^T\tilde{y},&space;\sigma^2&space;A^{-1})&space;\\&space;\sigma^2&space;&|\quad\tilde{y},&space;\beta,\tau_1^2,\dots,\tau_p^2&space;&\sim&&space;InvGamma\Big(\frac{1}{2}(n-1&plus;p),&space;\frac{1}{2}&space;\big((\tilde{y}-X\beta)^T&space;(\tilde{y}-X\beta)&plus;&space;\beta^TD_\tau^{-1}\beta\big)\Big)&space;\\&space;1/\tau_j^2&space;&|\quad&space;\tilde{y},&space;\beta,&space;\tau_{-j}^2&space;&\sim&&space;InvGauss\Big(\mu'=\sqrt{\frac{\lambda^2\sigma^2}{\beta_j^2}},&space;\lambda'=\lambda^2&space;\Big)&space;\end{array}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{array}{llcl}&space;&&&A&space;=&space;X^TX&space;&plus;&space;D_\tau^{-1}&space;\\&space;&&&D_\tau&space;=&space;diag(\tau_1^2,\dots,\tau_p^2)&space;\\&space;\\&space;\beta&space;&|&space;\quad&space;\tilde{y},&space;\sigma^2,&space;\tau_1^2,&space;\dots,&space;\tau_p^2&space;&\sim&&space;\mathcal{N}(A^{-1}&space;X^T\tilde{y},&space;\sigma^2&space;A^{-1})&space;\\&space;\sigma^2&space;&|\quad\tilde{y},&space;\beta,\tau_1^2,\dots,\tau_p^2&space;&\sim&&space;InvGamma\Big(\frac{1}{2}(n-1&plus;p),&space;\frac{1}{2}&space;\big((\tilde{y}-X\beta)^T&space;(\tilde{y}-X\beta)&plus;&space;\beta^TD_\tau^{-1}\beta\big)\Big)&space;\\&space;1/\tau_j^2&space;&|\quad&space;\tilde{y},&space;\beta,&space;\tau_{-j}^2&space;&\sim&&space;InvGauss\Big(\mu'=\sqrt{\frac{\lambda^2\sigma^2}{\beta_j^2}},&space;\lambda'=\lambda^2&space;\Big)&space;\end{array}" title="\begin{array}{llcl} &&&A = X^TX + D_\tau^{-1} \\ &&&D_\tau = diag(\tau_1^2,\dots,\tau_p^2) \\ \\ \beta &| \quad \tilde{y}, \sigma^2, \tau_1^2, \dots, \tau_p^2 &\sim& \mathcal{N}(A^{-1} X^T\tilde{y}, \sigma^2 A^{-1}) \\ \sigma^2 &|\quad\tilde{y}, \beta,\tau_1^2,\dots,\tau_p^2 &\sim& InvGamma\Big(\frac{1}{2}(n-1+p), \frac{1}{2} \big((\tilde{y}-X\beta)^T (\tilde{y}-X\beta)+ \beta^TD_\tau^{-1}\beta\big)\Big) \\ 1/\tau_j^2 &|\quad \tilde{y}, \beta, \tau_{-j}^2 &\sim& InvGauss\Big(\mu'=\sqrt{\frac{\lambda^2\sigma^2}{\beta_j^2}}, \lambda'=\lambda^2 \Big) \end{array}" /></a>

Cooling down of the posterior marginals can be achieved by a parameter shift of the distributions. For the Inverse Gaussian distribution this requires representing it as a <a href="https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution">Generalized Inverse Gaussian</a>.




# Status
In Progress

# References

[1] <a href="https://www.tandfonline.com/doi/abs/10.1198/016214508000000337">Park, Trevor, and George Casella. "The bayesian lasso." Journal of the American Statistical Association 103.482 (2008): 681-686.</a>
