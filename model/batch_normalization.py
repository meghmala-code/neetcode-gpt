import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        n=len(x)
        X=np.array(x)
        gamma= np.array(gamma)
        beta=np.array(beta)
        r_mean= np.array(running_mean)
        r_var= np.array(running_var)
        if training== True:
            mean= np.mean(X,axis=0)
            var= np.var(X,axis=0)
            x_hat= (X-mean)/np.sqrt(var + eps)
            #y= gamma*x_hat + beta
            r_mean=(1-momentum)*r_mean + momentum*mean
            r_var= (1-momentum)*r_var + momentum*var
        else:
            x_hat= (X-r_mean)/np.sqrt(r_var + eps)
        y= gamma*x_hat + beta
        return (
            np.round(y,4).tolist(),
            np.round(r_mean,4).tolist(),
            np.round(r_var,4).tolist()
        )
        pass
