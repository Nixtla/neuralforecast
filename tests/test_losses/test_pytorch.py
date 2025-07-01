import matplotlib.pyplot as plt
import torch
from fastcore.test import test_eq as _test_eq

from neuralforecast.losses.pytorch import (
    GMM,
    MAE,
    NBMM,
    PMM,
    DistributionLoss,
    HuberIQLoss,
    IQLoss,
    MQLoss,
)


# Unit tests to check MQLoss' stored quantiles
# attribute is correctly instantiated
def test_MQLoss_level():
    check = MQLoss(level=[80, 90])
    _test_eq(len(check.quantiles), 5)

def test_MQLoss_quantiles():
    check = MQLoss(quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900])
    print(check.output_names)
    print(check.quantiles)
    _test_eq(len(check.quantiles), 5)

    check = MQLoss(quantiles=[0.0100, 0.1000, 0.9000, 0.9900])
    _test_eq(len(check.quantiles), 4)


# Unit tests
# Check that default quantile is set to 0.5 at initialization
def test_IQLoss_default_quantile():
    check = IQLoss()
    _test_eq(check.q, 0.5)

# Check that quantiles are correctly updated - prediction
def test_IQLoss_update_quantile():
    check = IQLoss()
    check.update_quantile([0.7])
    _test_eq(check.q, 0.7)


# Unit tests to check DistributionLoss' stored quantiles
# attribute is correctly instantiated
def test_DistributionLoss_level():
    check = DistributionLoss(distribution="Normal", level=[80, 90])
    _test_eq(len(check.quantiles), 5)

def test_DistributionLoss_quantiles():
    check = DistributionLoss(
        distribution="Normal", quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900]
    )
    print(check.output_names)
    print(check.quantiles)
    _test_eq(len(check.quantiles), 5)

    check = DistributionLoss(
        distribution="Normal", quantiles=[0.0100, 0.1000, 0.9000, 0.9900]
    )
    _test_eq(len(check.quantiles), 4)

# Unit tests to check DistributionLoss' horizon weight
def test_DistributionLoss_horizon_weight():
    batch_size, horizon, n_outputs = 10, 3, 2
    y_hat = torch.rand(batch_size, horizon, n_outputs).chunk(2, dim=-1)
    y = torch.rand(batch_size, horizon, 1)
    y_loc = torch.rand(batch_size, 1, 1)
    y_scale = torch.rand(batch_size, 1, 1)

    loss = DistributionLoss(distribution="Normal", level=[80, 90])
    loss_with_hweights = DistributionLoss(
        distribution="Normal", level=[80, 90], horizon_weight=torch.ones(horizon)
    )

    distr_args = loss.scale_decouple(y_hat, y_loc, y_scale)
    distr_args_weighted = loss_with_hweights.scale_decouple(y_hat, y_loc, y_scale)

    _test_eq(loss(y, distr_args), loss_with_hweights(y, distr_args_weighted))


# Unit tests to check PMM's stored quantiles
# attribute is correctly instantiated
def test_PMM_level():
    check = PMM(n_components=2, level=[80, 90])
    _test_eq(len(check.quantiles), 5)

def test_PMM_quantiles():
    check = PMM(n_components=2, quantiles=[0.0100, 0.1000, 0.5, 0.9000, 0.9900])
    print(check.output_names)
    print(check.quantiles)
    _test_eq(len(check.quantiles), 5)

    check = PMM(n_components=2, quantiles=[0.0100, 0.1000, 0.9000, 0.9900])
    _test_eq(len(check.quantiles), 4)

# Create single mixture and broadcast to N,H,1,K
weights = torch.ones((1, 3))[None, :, :].unsqueeze(2)
lambdas = torch.Tensor([[5, 10, 15], [10, 20, 30]])[None, :, :].unsqueeze(2)

# Create repetitions for the batch dimension N.
N = 2
weights = torch.repeat_interleave(input=weights, repeats=N, dim=0)
lambdas = torch.repeat_interleave(input=lambdas, repeats=N, dim=0)

print("weights.shape (N,H,1,K) \t", weights.shape)
print("lambdas.shape (N,H,1, K) \t", lambdas.shape)

distr = PMM(quantiles=[0.1, 0.40, 0.5, 0.60, 0.9], weighted=True)
weights = torch.ones_like(lambdas)
distr_args = (lambdas, weights)
samples, sample_mean, quants = distr.sample(distr_args)

print("samples.shape (N,H,1,num_samples) ", samples.shape)
print("sample_mean.shape (N,H,1,1) ", sample_mean.shape)
print("quants.shape  (N,H,1,Q) \t\t", quants.shape)

# Plot synthethic data
x_plot = range(quants.shape[1])  # H length
y_plot_hat = quants[0, :, 0, :]  # Filter N,G,T -> H,Q
samples_hat = samples[0, :, 0, :]  # Filter N,G,T -> H,num_samples

# Kernel density plot for single forecast horizon \tau = t+1
fig, ax = plt.subplots(figsize=(3.7, 2.9))

ax.hist(samples_hat[0, :], alpha=0.5, label=r"Horizon $\tau+1$")
ax.hist(samples_hat[1, :], alpha=0.5, label=r"Horizon $\tau+2$")
ax.set(xlabel="Y values", ylabel="Probability")
plt.title("Single horizon Distributions")
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
plt.grid()
plt.show()
plt.close()

# Plot simulated trajectory
fig, ax = plt.subplots(figsize=(3.7, 2.9))
plt.plot(x_plot, y_plot_hat[:, 2], color="black", label="median [q50]")
plt.fill_between(
    x_plot,
    y1=y_plot_hat[:, 1],
    y2=y_plot_hat[:, 3],
    facecolor="blue",
    alpha=0.4,
    label="[p25-p75]",
)
plt.fill_between(
    x_plot,
    y1=y_plot_hat[:, 0],
    y2=y_plot_hat[:, 4],
    facecolor="blue",
    alpha=0.2,
    label="[p1-p99]",
)
ax.set(xlabel="Horizon", ylabel="Y values")
plt.title("PMM Probabilistic Predictions")
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
plt.grid()
plt.show()
plt.close()



# Create single mixture and broadcast to N,H,1,K
means = torch.Tensor([[5, 10, 15], [10, 20, 30]])[None, :, :].unsqueeze(2)

# # Create repetitions for the batch dimension N.
N = 2
means = torch.repeat_interleave(input=means, repeats=N, dim=0)
weights = torch.ones_like(means)
stds = torch.ones_like(means)

print("weights.shape (N,H,1,K) \t", weights.shape)
print("means.shape (N,H,1,K) \t", means.shape)
print("stds.shape (N,H,1,K) \t", stds.shape)

distr = GMM(quantiles=[0.1, 0.40, 0.5, 0.60, 0.9], weighted=True)
distr_args = (means, stds, weights)
samples, sample_mean, quants = distr.sample(distr_args)

print("samples.shape (N,H,1,num_samples) ", samples.shape)
print("sample_mean.shape (N,H,1,1) ", sample_mean.shape)
print("quants.shape  (N,H,1, Q) \t\t", quants.shape)

# Plot synthethic data
x_plot = range(quants.shape[1])  # H length
y_plot_hat = quants[0, :, 0, :]  # Filter N,G,T -> H,Q
samples_hat = samples[0, :, 0, :]  # Filter N,G,T -> H,num_samples

# Kernel density plot for single forecast horizon \tau = t+1
fig, ax = plt.subplots(figsize=(3.7, 2.9))

ax.hist(samples_hat[0, :], alpha=0.5, bins=50, label=r"Horizon $\tau+1$")
ax.hist(samples_hat[1, :], alpha=0.5, bins=50, label=r"Horizon $\tau+2$")
ax.set(xlabel="Y values", ylabel="Probability")
plt.title("Single horizon Distributions")
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
plt.grid()
plt.show()
plt.close()

# Plot simulated trajectory
fig, ax = plt.subplots(figsize=(3.7, 2.9))
plt.plot(x_plot, y_plot_hat[:, 2], color="black", label="median [q50]")
plt.fill_between(
    x_plot,
    y1=y_plot_hat[:, 1],
    y2=y_plot_hat[:, 3],
    facecolor="blue",
    alpha=0.4,
    label="[p25-p75]",
)
plt.fill_between(
    x_plot,
    y1=y_plot_hat[:, 0],
    y2=y_plot_hat[:, 4],
    facecolor="blue",
    alpha=0.2,
    label="[p1-p99]",
)
ax.set(xlabel="Horizon", ylabel="Y values")
plt.title("GMM Probabilistic Predictions")
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
plt.grid()
plt.show()
plt.close()


# Create single mixture and broadcast to N,H,1,K
counts = torch.Tensor([[5, 10, 15], [10, 20, 30]])[None, :, :].unsqueeze(2)

# # Create repetitions for the batch dimension N.
N = 2
counts = torch.repeat_interleave(input=counts, repeats=N, dim=0)
weights = torch.ones_like(counts)
probs = torch.ones_like(counts) * 0.5

print("weights.shape (N,H,1,K) \t", weights.shape)
print("counts.shape (N,H,1,K) \t", counts.shape)
print("probs.shape (N,H,1,K) \t", probs.shape)

model = NBMM(quantiles=[0.1, 0.40, 0.5, 0.60, 0.9], weighted=True)
distr_args = (counts, probs, weights)
samples, sample_mean, quants = model.sample(distr_args, num_samples=2000)

print("samples.shape (N,H,1,num_samples) ", samples.shape)
print("sample_mean.shape (N,H,1,1) ", sample_mean.shape)
print("quants.shape  (N,H,1,Q) \t\t", quants.shape)

# Plot synthethic data
x_plot = range(quants.shape[1])  # H length
y_plot_hat = quants[0, :, 0, :]  # Filter N,G,T -> H,Q
samples_hat = samples[0, :, 0, :]  # Filter N,G,T -> H,num_samples

# Kernel density plot for single forecast horizon \tau = t+1
fig, ax = plt.subplots(figsize=(3.7, 2.9))

ax.hist(samples_hat[0, :], alpha=0.5, bins=30, label=r"Horizon $\tau+1$")
ax.hist(samples_hat[1, :], alpha=0.5, bins=30, label=r"Horizon $\tau+2$")
ax.set(xlabel="Y values", ylabel="Probability")
plt.title("Single horizon Distributions")
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
plt.grid()
plt.show()
plt.close()

# Plot simulated trajectory
fig, ax = plt.subplots(figsize=(3.7, 2.9))
plt.plot(x_plot, y_plot_hat[:, 2], color="black", label="median [q50]")
plt.fill_between(
    x_plot,
    y1=y_plot_hat[:, 1],
    y2=y_plot_hat[:, 3],
    facecolor="blue",
    alpha=0.4,
    label="[p25-p75]",
)
plt.fill_between(
    x_plot,
    y1=y_plot_hat[:, 0],
    y2=y_plot_hat[:, 4],
    facecolor="blue",
    alpha=0.2,
    label="[p1-p99]",
)
ax.set(xlabel="Horizon", ylabel="Y values")
plt.title("NBM Probabilistic Predictions")
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=1)
plt.grid()
plt.show()
plt.close()


# Unit tests
# Check that default quantile is set to 0.5 at initialization
def test_HuberIQLoss_init():
    check = HuberIQLoss()
    _test_eq(check.q, 0.5)

# Check that quantiles are correctly updated - prediction
def test_HuberIQLoss_update_quantile():
    check = HuberIQLoss()
    check.update_quantile([0.7])
    _test_eq(check.q, 0.7)


# Each 1 is an error, there are 6 datapoints.
def test_MAE_complete_mask():
    y = torch.Tensor([[0, 0, 0], [0, 0, 0]]).unsqueeze(-1)
    y_hat = torch.Tensor([[0, 0, 1], [1, 0, 1]]).unsqueeze(-1)

    # Complete mask and horizon_weight
    mask = torch.Tensor([[1, 1, 1], [1, 1, 1]]).unsqueeze(-1)
    horizon_weight = torch.Tensor([1, 1, 1])

    mae = MAE(horizon_weight=horizon_weight)
    loss = mae(y=y, y_hat=y_hat, mask=mask)
    assert loss == (3 / 6), "Should be 3/6"


# Incomplete mask and complete horizon_weight
def test_MAE_incomplete_mask():
    # Only 1 error and points is masked.
    y = torch.Tensor([[0, 0, 0], [0, 0, 0]]).unsqueeze(-1)
    y_hat = torch.Tensor([[0, 0, 1], [1, 0, 1]]).unsqueeze(-1)

    mask = torch.Tensor([[1, 1, 1], [0, 1, 1]]).unsqueeze(-1)
    horizon_weight = torch.Tensor([1, 1, 1])
    mae = MAE(horizon_weight=horizon_weight)
    loss = mae(y=y, y_hat=y_hat, mask=mask)
    assert loss == (2 / 5), "Should be 2/5"

    # Complete mask and incomplete horizon_weight
    mask = torch.Tensor([[1, 1, 1], [1, 1, 1]]).unsqueeze(-1)
    horizon_weight = torch.Tensor([1, 1, 0])  # 2 errors and points are masked.
    mae = MAE(horizon_weight=horizon_weight)
    loss = mae(y=y, y_hat=y_hat, mask=mask)
    assert loss == (1 / 4), "Should be 1/4"

    # Incomplete mask and incomplete horizon_weight
    mask = torch.Tensor([[0, 1, 1], [1, 1, 1]]).unsqueeze(-1)
    horizon_weight = torch.Tensor([1, 1, 0])  # 2 errors are masked, and 3 points.
    mae = MAE(horizon_weight=horizon_weight)
    loss = mae(y=y, y_hat=y_hat, mask=mask)
    assert loss == (1 / 3), "Should be 1/3"
