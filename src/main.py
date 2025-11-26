from config import CONFIG
from data_generation import generate_paths
from pricing import price_swaption, compute_swap_gradients, swaption_price_and_grads, compute_swap_rate
from networks import ActionNetwork, PRICE_NETWORK
from utils import compute_payoff
from training import train_model, EarlyStopping
from evaluation import evaluate_model
import torch.optim as optim
import torch
from numpy import round
import copy
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Set seed
torch.manual_seed(CONFIG["seed"])


# Define hedging portfolio
swap1_spec = {'forward_start': CONFIG["T_alpha"], 'tenor': CONFIG["length_tenor"]}
swap2_spec = {'forward_start': 120, 'tenor': 24}
swaps_expiries = [swap1_spec['forward_start'], swap2_spec['forward_start']]
# Data generation
factorpath_IS, short_rate_IS, factorpath_OOS, short_rate_OOS = generate_paths(CONFIG)

# Pricing
_, _, zc_price_IS = price_swaption(factorpath_IS, CONFIG)
_, _, zc_price_OOS = price_swaption(factorpath_OOS, CONFIG)

# swaps par rate
swap1_par_rate = compute_swap_rate(factorpath_IS, CONFIG, swap1_spec['forward_start'], swap1_spec['tenor'])
swap2_par_rate = compute_swap_rate(factorpath_IS, CONFIG, swap2_spec['forward_start'], swap2_spec['tenor'])


swap1_price_IS, _ = compute_swap_gradients(factorpath_IS, CONFIG, round(swap1_par_rate,6),
                                           swap1_spec['forward_start'], swap1_spec['tenor'])
swap2_price_IS, _ = compute_swap_gradients(factorpath_IS, CONFIG, round(swap2_par_rate, 6),
                                           swap2_spec['forward_start'], swap2_spec['tenor'])

swap1_par_rate_oos = compute_swap_rate(factorpath_OOS, CONFIG, swap1_spec['forward_start'], swap1_spec['tenor'])
swap2_par_rate_oos = compute_swap_rate(factorpath_OOS, CONFIG, swap2_spec['forward_start'], swap2_spec['tenor'])

swap1_price_OOS, _ = compute_swap_gradients(factorpath_OOS, CONFIG, round(swap1_par_rate_oos,6),
                                           swap1_spec['forward_start'], swap1_spec['tenor'])
swap2_price_OOS, _ = compute_swap_gradients(factorpath_OOS, CONFIG, round(swap2_par_rate_oos, 6),
                                           swap2_spec['forward_start'], swap2_spec['tenor'])


CONFIG["fixed_rate"] = round(swap1_par_rate,6)
payoff_IS = compute_payoff(zc_price_IS, CONFIG["fixed_rate"], CONFIG["dt"], CONFIG["notional_value"],
                           CONFIG["swaptype"])
payoff_OOS = compute_payoff(zc_price_OOS, CONFIG["fixed_rate"], CONFIG["dt"], CONFIG["notional_value"],
                           CONFIG["swaptype"])

mlp_model = PRICE_NETWORK(
    insize=CONFIG["input_size_mlp"],
    outsize=CONFIG["output_size_mlp"],
    HLsizes=CONFIG["hidden_size_mlp"]
)

# Load the saved weights
KAN_MODEL_PATH = os.path.join(ROOT, "..", "models", "pretrained", CONFIG["network_swaption_pricing"])
mlp_model.load_state_dict(torch.load(KAN_MODEL_PATH))
mlp_model.eval()

swaption_price_IS, _ = swaption_price_and_grads(factorpath_IS, CONFIG, mlp_model, isgrad=False)
swaption_price_OOS, _ = swaption_price_and_grads(factorpath_OOS, CONFIG, mlp_model, isgrad=False)

data_IS = {
    "swap1_price": swap1_price_IS,
    "swap2_price": swap2_price_IS,
    "zc_price": zc_price_IS,
    "short_rate": short_rate_IS,
    "factors": factorpath_IS,
    "swaption_price": swaption_price_IS,
    "payoff": payoff_IS,
    "swaps_expiries": swaps_expiries
}

data_OOS = {
    "swap1_price": swap1_price_OOS,
    "swap2_price": swap2_price_OOS,
    "zc_price": zc_price_OOS,
    "short_rate": short_rate_OOS,
    "factors": factorpath_OOS,
    "swaption_price": swaption_price_OOS,
    "payoff": payoff_OOS,
    "swaps_expiries": swaps_expiries
}

penalty_methods = ["mse"]  # ["mse", "cvar", "downside]
for method in penalty_methods:
    print(f"\n==== Training with penalty method: {method} ====")

    config = copy.deepcopy(CONFIG)
    config["penalty_method"] = method
    net_name = f"action_network_2swaps_{method}.pth"
    config["network_path"] = os.path.join(ROOT, "..", "models", "final_trained", net_name)

    model = ActionNetwork(config["input_size"], config["hidden_size"], config["output_size"],
                          dropout_prob=config["dropout_prob"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, min_lr=1e-4, patience=5)
    early_stopping = EarlyStopping(patience=config["patience"], min_delta=0.0, path=config["network_path"])

    for epoch in range(config["nepochs"]):
        model.train()
        loss = train_model(model, optimizer, data_IS, config)
        _, loss_valid, *_ = evaluate_model(model, data_OOS, config)
        scheduler.step(loss_valid)
        print(
            f"Epoch {epoch + 1}/{config['nepochs']}, Loss train: {loss.item():.8f}, Loss validation: {loss_valid:.8f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        early_stopping(loss_valid, model)
        if early_stopping.earlystop:
            print(f"Early stopping at epoch {epoch + 1}, best loss: {early_stopping.best_value:.6f}")
            break
