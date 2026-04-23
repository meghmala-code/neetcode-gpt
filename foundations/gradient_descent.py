class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learlearningate * f'(x)
        # Round final answer to 5 decimal places
        x=init
        for i in range(0, iterations):
            f= pow(x, 2)
            df= 2*x
            x= x - learning_rate*df
        ans= round (x, 5)
        return(ans)
        pass
