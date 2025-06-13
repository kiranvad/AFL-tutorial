import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import warnings
warnings.filterwarnings('ignore')

class CostConstrainedBO:
    def __init__(self, objective_func, cost_func, bounds, budget, kernel=None):
        self.objective_func = objective_func
        self.cost_func = cost_func
        self.bounds = bounds
        self.budget = budget
        self.dimension = len(bounds)
        
        if kernel is None:
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        
        self.gp_obj = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, 
                                               normalize_y=True, n_restarts_optimizer=5)
        self.gp_cost = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, 
                                                normalize_y=True, n_restarts_optimizer=5)
        
        self.X_observed = []
        self.y_observed = []
        self.costs_observed = []
        self.total_cost = 0.0
        
    def _normalize_bounds(self, x):
        """Normalize point to [0,1]^d"""
        x_norm = np.zeros_like(x)
        for i in range(len(x)):
            x_norm[i] = (x[i] - self.bounds[i][0]) / (self.bounds[i][1] - self.bounds[i][0])
        return x_norm
    
    def _denormalize_bounds(self, x_norm):
        """Denormalize point from [0,1]^d to original bounds"""
        x = np.zeros_like(x_norm)
        for i in range(len(x_norm)):
            x[i] = x_norm[i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
        return x
    
    def add_observation(self, x, y, cost):
        """Add new observation"""
        self.X_observed.append(x)
        self.y_observed.append(y)
        self.costs_observed.append(cost)
        self.total_cost += cost
        
    def fit_gps(self):
        """Fit Gaussian processes to observations"""
        if len(self.X_observed) == 0:
            return
            
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        costs = np.array(self.costs_observed)
        
        self.gp_obj.fit(X, y)
        # Use log-transformed cost for GP (as mentioned in paper)
        self.gp_cost.fit(X, np.log(costs + 1e-8))
    
    def predict_objective(self, X):
        """Predict objective mean and std"""
        if len(self.X_observed) == 0:
            return np.zeros(len(X)), np.ones(len(X))
        mean, std = self.gp_obj.predict(X, return_std=True)
        return mean, std
    
    def predict_cost(self, X):
        """Predict cost"""
        if len(self.X_observed) == 0:
            return np.array([self.cost_func(x) for x in X])
        log_mean, log_std = self.gp_cost.predict(X, return_std=True)
        return np.exp(log_mean)
    
    def expected_improvement(self, X):
        """Calculate Expected Improvement"""
        if len(self.X_observed) == 0:
            return np.ones(len(X))
        
        mu, sigma = self.predict_objective(X)
        sigma = np.maximum(sigma, 1e-9)
        
        f_best = np.min(self.y_observed)
        
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei
    
    def ei_per_unit_cost(self, X):
        """Calculate EI per unit cost"""
        ei = self.expected_improvement(X)
        costs = self.predict_cost(X)
        costs = np.maximum(costs, 1e-9)  # Prevent division by zero
        return ei / costs
    
    def optimize_acquisition(self, acquisition_func, n_restarts=10):
        """Optimize acquisition function"""
        def objective(x_norm):
            x = self._denormalize_bounds(x_norm)
            return -acquisition_func(np.array([x]))[0]
        
        best_x = None
        best_val = np.inf
        
        # Try multiple random starting points
        for _ in range(n_restarts):
            x0_norm = np.random.random(self.dimension)
            try:
                result = minimize(objective, x0_norm, method='L-BFGS-B', 
                                bounds=[(0, 1)] * self.dimension)
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = self._denormalize_bounds(result.x)
            except:
                continue
                
        return best_x if best_x is not None else self._denormalize_bounds(np.random.random(self.dimension))
    
    def rollout_policy(self, x_start, horizon=2, n_samples=50):
        """Simulate rollout policy"""
        if self.total_cost + self.cost_func(x_start) > self.budget:
            return -np.inf  # Infeasible
        
        total_reward = 0.0
        n_feasible = 0
        
        for _ in range(n_samples):
            # Simulate trajectory
            current_x = x_start.copy()
            current_cost = self.total_cost
            reward = 0.0
            feasible = True
            
            for h in range(horizon):
                # Check if we can afford this evaluation
                eval_cost = self.cost_func(current_x)
                if current_cost + eval_cost > self.budget:
                    feasible = False
                    break
                
                # Simulate objective value
                if len(self.X_observed) > 0:
                    mu, sigma = self.predict_objective(np.array([current_x]))
                    y_sim = np.random.normal(mu[0], sigma[0])
                else:
                    y_sim = self.objective_func(current_x)
                
                # Calculate reward (improvement over current best)
                if len(self.y_observed) > 0:
                    current_best = np.min(self.y_observed)
                else:
                    current_best = np.inf
                    
                improvement = max(0, current_best - y_sim)
                reward += improvement
                current_cost += eval_cost
                
                # For next iteration, use EIpu (except last step uses EI)
                if h < horizon - 1:
                    # Use EIpu for intermediate steps
                    try:
                        current_x = self.optimize_acquisition(self.ei_per_unit_cost)
                    except:
                        current_x = self._denormalize_bounds(np.random.random(self.dimension))
                
            if feasible:
                total_reward += reward
                n_feasible += 1
        
        return total_reward / max(1, n_feasible)
    
    def rollout_acquisition(self, X, horizon=2):
        """Calculate rollout acquisition values"""
        values = []
        for x in X:
            val = self.rollout_policy(x, horizon)
            values.append(val)
        return np.array(values)

def synthetic_objective(x):
    """Synthetic objective function from Figure 4"""
    r = np.linalg.norm(x)
    return 10 * r * np.sin(2 * np.pi * r)

def synthetic_cost(x):
    """Synthetic cost function from Figure 4"""
    r = np.linalg.norm(x)
    return 10 - 5 * r

def run_bo_experiment(method, n_iterations=20, n_runs=50):
    """Run Bayesian optimization experiment"""
    bounds = [(-1, 1), (-1, 1)]
    budget = 150
    
    results = []
    
    for run in range(n_runs):
        np.random.seed(run)  # For reproducibility
        
        bo = CostConstrainedBO(synthetic_objective, synthetic_cost, bounds, budget)
        
        # Initialize with a few random points
        for _ in range(3):
            x_init = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            y_init = synthetic_objective(x_init)
            cost_init = synthetic_cost(x_init)
            
            if bo.total_cost + cost_init <= budget:
                bo.add_observation(x_init, y_init, cost_init)
        
        run_results = []
        costs_used = [bo.total_cost]
        best_values = [np.min(bo.y_observed) if bo.y_observed else np.inf]
        
        # Run BO iterations
        iteration = 0
        while bo.total_cost < budget and iteration < n_iterations:
            bo.fit_gps()
            
            # Generate candidate points
            n_candidates = 1000
            X_candidates = np.array([[np.random.uniform(b[0], b[1]) for b in bounds] 
                                   for _ in range(n_candidates)])
            
            # Select next point based on method
            if method == 'EI':
                acq_values = bo.expected_improvement(X_candidates)
            elif method == 'EIpu':
                acq_values = bo.ei_per_unit_cost(X_candidates)
            elif method.startswith('Rollout'):
                horizon = int(method.split()[-1])
                acq_values = bo.rollout_acquisition(X_candidates, horizon)
            
            # Find feasible candidates
            feasible_mask = np.array([bo.total_cost + synthetic_cost(x) <= budget 
                                    for x in X_candidates])
            
            if not np.any(feasible_mask):
                break
                
            feasible_indices = np.where(feasible_mask)[0]
            feasible_acq = acq_values[feasible_indices]
            
            best_idx = feasible_indices[np.argmax(feasible_acq)]
            x_next = X_candidates[best_idx]
            
            # Evaluate
            y_next = synthetic_objective(x_next)
            cost_next = synthetic_cost(x_next)
            
            bo.add_observation(x_next, y_next, cost_next)
            
            costs_used.append(bo.total_cost)
            best_values.append(np.min(bo.y_observed))
            
            iteration += 1
        
        results.append((costs_used, best_values))
    
    return results

# Create the plots
def plot_figure4():
    """Reproduce Figure 4 from the paper"""
    
    # Create subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Objective and Cost functions
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate objective and cost on grid
    obj_vals = np.zeros_like(X)
    cost_vals = np.zeros_like(X)
    
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([X[i,j], Y[i,j]])
            obj_vals[i,j] = synthetic_objective(point)
            cost_vals[i,j] = synthetic_cost(point)
    
    # Plot radial slice at y=0
    r_slice = np.linspace(-1, 1, 200)
    obj_slice = [synthetic_objective(np.array([r, 0])) for r in r_slice]
    cost_slice = [synthetic_cost(np.array([r, 0])) for r in r_slice]
    
    ax1.plot(r_slice, obj_slice, 'b-', label='f(x)', linewidth=2)
    ax1.plot(r_slice, cost_slice, 'r-', label='c(x)', linewidth=2)
    ax1.set_xlabel('Radial slice r')
    ax1.set_ylabel('Objective value')
    ax1.set_title('Objective and Cost')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: BO Performance comparison
    print("Running BO experiments...")
    methods = ['EI', 'EIpu', 'Rollout 2', 'Rollout 4']
    colors = ['blue', 'green', 'pink', 'red']
    
    for method, color in zip(methods, colors):
        print(f"Running {method}...")
        results = run_bo_experiment(method, n_runs=10)  # Reduced for faster execution
        
        # Interpolate results to common cost grid
        cost_grid = np.linspace(0, 150, 100)
        interpolated_values = []
        
        for costs_used, best_values in results:
            # Convert to percentage of budget used
            cost_pct = np.array(costs_used) / 150 * 100
            interp_values = np.interp(cost_grid / 150 * 100, cost_pct, best_values)
            interpolated_values.append(interp_values)
        
        interpolated_values = np.array(interpolated_values)
        mean_values = np.mean(interpolated_values, axis=0)
        std_values = np.std(interpolated_values, axis=0)
        
        cost_pct_grid = cost_grid / 150 * 100
        ax2.plot(cost_pct_grid, mean_values, color=color, label=method, linewidth=2)
        ax2.fill_between(cost_pct_grid, mean_values - std_values, 
                        mean_values + std_values, color=color, alpha=0.2)
    
    ax2.set_xlabel('% Cost Used')
    ax2.set_ylabel('Best Objective Value')
    ax2.set_title('BO Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_figure4()