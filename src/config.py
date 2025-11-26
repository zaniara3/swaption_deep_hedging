CONFIG = {
    "seed": 2022,
    "dt": 1 / 12, # time step in year
    "num_paths": 10000, # number of Monte Carlo paths
    "T_alpha": 60,  # Swaption maturity
    "length_tenor": 120,  # Underlying Swap tenor
    "fixed_rate": 0.025083, # fixed rate
    "notional_value": 1.0, # notional value
    "swaptype": "payer",   # "payer" or "receiver"
    # Model parameters of swaption pricing network
    "hidden_size_mlp": [16, 32, 16],
    "input_size_mlp": 4,
    "output_size_mlp": 1,
    "learning_rate_mlp": 0.001,
    "network_swaption_pricing": "swaption_network_al60_be180_k25083_forgrad_fastkan.pth",
    # Model parameters of replicating portfolio optimization network
    "hidden_size": [8, 32, 32, 8],
    "input_size": 5,
    "output_size": 2,
    # "max_leverage": 24.0,
    "dropout_prob": 0.0,
    "nepochs": 800,
    "minibatchsize": 2048,
    "learning_rate": 0.005,
    "patience": 200,
    "tc_multiplier": 0.0,
    # Objective function type
    "penalty_method": "mse",  # "mse", "cvar", "downside"
    "network_path": "",
    # Constraints parameters on hedge ratios
    "per_swap_leverage": 2,
    "portfolio_limit": 3,
    "budget_cap": 1,
    "dynamic_basis": True,      # basis = |V_t| (True) or |V_0| (False)
}