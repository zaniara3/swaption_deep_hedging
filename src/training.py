import torch
from utils import penalty_function, prepare_inputs, apply_leverage_constraints
import torch.optim as optim
import numpy as np


class EarlyStopping:
    def __init__(self, patience, min_delta=0.0, path='model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.wait = 0
        self.best_value = float('inf')
        self.earlystop = False

    def __call__(self, value, model):
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.wait = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.earlystop = True


def train_model(model, optimizer, data, config):
    loss_train = torch.tensor(float('inf'))
    minibatchsize = config["minibatchsize"]
    niter = data["swap1_price"].shape[0] // minibatchsize

    for it in range(niter):
        idx = slice(it * minibatchsize, (it + 1) * minibatchsize)
        swap1, swap2 = torch.tensor(data["swap1_price"][idx]), torch.tensor(data["swap2_price"][idx])
        zc = torch.tensor(data["zc_price"][idx])
        short = torch.tensor(data["short_rate"][idx])
        factors = torch.tensor(data["factors"][idx])
        payoff = torch.tensor(data["payoff"][idx])
        swaps_expiry = data["swaps_expiries"]

        port_val = torch.zeros((minibatchsize, config["T_alpha"] + 1))
        port_val[:, 0] = torch.tensor(data["swaption_price"][idx, 0])

        tc_multiplier = config["tc_multiplier"]

        input_buf = torch.zeros((minibatchsize, config["input_size"]))
        prev_weights = torch.zeros((minibatchsize, 2))
        for t in range(config["T_alpha"]):
            active_cols = torch.tensor([t < swaps_expiry[0], t < swaps_expiry[1]], dtype=bool)
            ttm = (config["T_alpha"] - t) * config["dt"]
            inputs = prepare_inputs(input_buf.numpy(), factors[:, t].detach().numpy(), port_val[:, t].detach().numpy(),
                                    ttm)
            prices_t = torch.stack([torch.abs(swap1[:, t]),
                                    torch.abs(swap2[:, t])], dim=1)  # (N,3)

            # Basis: dynamic = |V_t|, static = |V_0|
            if config.get("dynamic_basis", True):
                basis_t = torch.abs(port_val[:, t]) + 1e-8  # (N,)
            else:
                basis_t = torch.abs(port_val[:, 0]) + 1e-8
            weights = model(inputs)

            weights = apply_leverage_constraints(
            weights=weights,
            prices=prices_t,
            basis=basis_t,
            per_swap_leverage=config.get("per_swap_leverage", 1.5),
            portfolio_limit=config.get("portfolio_limit", 3.0),
            budget_cap=config.get("budget_cap", 1.0),
            active_swaps=active_cols,
            iters=40,
            eps=1e-12)

            if t > 0:
                transaction_cost = (torch.abs(weights[:, 0] - prev_weights[:, 0]) * torch.abs(swap1[:, t])
                                    + torch.abs(weights[:, 1] - prev_weights[:, 1]) * torch.abs(swap2[:, t])) * tc_multiplier
            else:
                transaction_cost = torch.zeros(minibatchsize)
            cash = (port_val[:, t] - weights[:, 0] * swap1[:, t] - weights[:, 1] * swap2[:, t]
                    - transaction_cost)
            port_val[:, t + 1] = (weights[:, 0] * swap1[:, t + 1] + weights[:, 1] * swap2[:, t + 1]
                                  + cash * torch.exp(short[:, t] * config["dt"]))
            prev_weights = weights.clone()

        hedge_error = payoff - port_val[:, -1]
        loss = penalty_function(hedge_error, config["penalty_method"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train = loss

    return loss_train


def data_preparation(data, config):
    X_supervised = []
    Y_supervised = []

    swap1, swap2 = torch.tensor(data["swap1_price"]), torch.tensor(data["swap2_price"])
    zc = torch.tensor(data["zc_price"])
    short = torch.tensor(data["short_rate"])
    factors = torch.tensor(data["factors"])
    payoff = torch.tensor(data["payoff"])
    target = torch.tensor(data["target_weights"])

    port_val_ = torch.zeros((config["num_paths"], config["T_alpha"] + 1))
    port_val_[:, 0] = torch.tensor(data["swaption_price"][:, 0])

    tc_multiplier = config["tc_multiplier"]

    input_buf = torch.zeros((config["num_paths"], config["input_size"]))
    prev_weights = torch.zeros((config["num_paths"], 3))
    T_alpha = int(config["T_alpha"])
    for t in range(T_alpha):
        ttm = (T_alpha - t) * config["dt"]
        inputs = prepare_inputs(input_buf.numpy(), factors[:, t].detach().numpy(), port_val_[:, t].detach().numpy(),
                                ttm)
        weights_ = target[:, t, 0:3]
        if t > 0:
            transaction_cost_ = (torch.abs(weights_[:, 0] - prev_weights[:, 0]) * torch.abs(swap1[:, t])
                                + torch.abs(weights_[:, 1] - prev_weights[:, 1]) * torch.abs(swap2[:, t])) * tc_multiplier
        else:
            transaction_cost_ = torch.zeros(config["num_paths"])
        cash_ = (port_val_[:, t] - weights_[:, 0] * swap1[:, t] - weights_[:, 1] * swap2[:, t]
                -  transaction_cost_)
        port_val_[:, t + 1] = (weights_[:, 0] * swap1[:, t + 1] + weights_[:, 1] * swap2[:, t + 1]
                            + cash_ * torch.exp(short[:, t] * config["dt"]))
        prev_weights = weights_.clone()

        X_supervised.append(inputs)
        Y_supervised.append(weights_)

    X_supervised = np.vstack(X_supervised)
    Y_supervised = np.vstack(Y_supervised)

    return X_supervised, Y_supervised
