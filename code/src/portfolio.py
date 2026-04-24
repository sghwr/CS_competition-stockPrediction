import torch
import torch.nn.functional as F


def calibrate_weights(scores, temperature=1.0, max_weight=0.4, top_k=5):
    if temperature <= 0:
        temperature = 1e-6
    scaled = scores / temperature
    weights = F.softmax(scaled, dim=-1)
    weights = torch.clamp(weights, max=max_weight)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return weights


def build_portfolio(scores, temperature=1.0, max_weight=0.4, top_k=5):
    batch_size, num_stocks = scores.shape
    device = scores.device

    topk_vals, topk_idx = torch.topk(scores, top_k, dim=-1)
    topk_weights = calibrate_weights(topk_vals, temperature, max_weight, top_k)

    portfolio = torch.zeros_like(scores)
    portfolio.scatter_(1, topk_idx, topk_weights)

    cash_residual = 1.0 - portfolio.sum(dim=-1, keepdim=True)
    return portfolio, cash_residual


def compute_final_score(pred_weights, actual_returns):
    EPS = 1e-8
    pred_sum = (pred_weights * actual_returns).sum(dim=-1)
    random_sum = actual_returns.mean(dim=-1)
    max_ret, _ = actual_returns.max(dim=-1)
    min_ret, _ = actual_returns.min(dim=-1)
    best_sum = max_ret
    worst_sum = min_ret
    denominator = best_sum - worst_sum
    score = (pred_sum - random_sum) / (denominator + EPS)
    return score
