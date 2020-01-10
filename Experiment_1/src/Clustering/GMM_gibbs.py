import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import scipy.stats as stats
from sklearn.metrics import adjusted_rand_score
from tqdm import trange

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--phn", type=Path, required=True)
parser.add_argument("--K", type=int, required=True)
parser.add_argument("--iter", type=int, required=True)

parser.add_argument("--print_all_ARI", action="store_true")
parser.add_argument("--print_ARI_span", type=int)

args = parser.parse_args()

source = np.load(args.source)
phn = np.load(args.phn)
K = args.K

keys = sorted(list(source.keys()))

datas = np.concatenate([source[key] for key in keys], axis=0)
labels = np.concatenate([phn[key] for key in keys], axis=0)
N, D = datas.shape


#%% hyper parameters.
alpha_0 = np.ones(K)
mu_0 = np.zeros((K, D))
lambda_0 = np.ones(K)
nu_0 = np.ones(K) * D
Psi_0 = np.empty((K, D, D))
for k in range(K):
    Psi_0[k] = np.identity(D)

#%% Local parameters.
pi = stats.dirichlet.rvs(alpha_0)[0]
mu = np.empty((K, D))
Sigma = np.empty((K, D, D))

for k in range(K):
    Sigma[k] = stats.invwishart.rvs(df=nu_0[k], scale=Psi_0[k])
    mu[k] = stats.multivariate_normal.rvs(mean=mu_0[k], cov=(Sigma[k] / lambda_0[k]))

#%% Temporary variables.
p_xi = np.empty((N, K))

#%% Latent variables
hidden_state = np.zeros(N, dtype=int)

for i in trange(args.iter):

    # Calculate phase.
    for k in range(K):
        p_xi[:,k] = stats.multivariate_normal.pdf(datas, mean=mu[k], cov=Sigma[k])
    p_xi *= pi

    # Resampling labels.
    for n in range(N):
        pi_ast = p_xi[n] / p_xi[n].sum()
        hidden_state[n] = np.random.choice(K, p=pi_ast)
    m = np.bincount(hidden_state, minlength=K)

    # Resampling pi.
    alpha_ast = alpha_0 + m
    pi[:] = stats.dirichlet.rvs(alpha_ast)

    # Resampling mu_k and Sigma_k
    for k in range(K):
        if m[k] == 0:
            data_k = np.matrix(np.zeros(D))
        else:
            data_k = np.matrix(datas[hidden_state == k])

        # Calculate Temporary variables.
        mean_of_data_k = data_k.mean(axis=0)
        S_k = np.dot((data_k - mean_of_data_k).T, data_k - mean_of_data_k)

        # Resampling Sigma_k.
        nu_ast = m[k] + nu_0[k]
        Psi_ast = S_k + Psi_0[k] + (m[k] * lambda_0[k])/(m[k] + lambda_0[k]) * np.dot((mean_of_data_k - mu_0[k]).T, mean_of_data_k - mu_0[k])
        Sigma[k] = stats.invwishart.rvs(df=nu_ast, scale=Psi_ast)

        # Resampling mu_k.
        lambda_ast = m[k] + lambda_0[k]
        mu_ast = ((m[k] * mean_of_data_k + lambda_0[k] * mu_0[k]) / (m[k] + lambda_0[k])).A1
        mu[k] = stats.multivariate_normal.rvs(mean=mu_ast, cov=(Sigma[k] / lambda_ast))

    if args.print_all_ARI or (args.print_ARI_span and i % args.print_ARI_span == 0):
        print(f"ARI: {adjusted_rand_score(hidden_state, labels)}")

print(f"ARI: {adjusted_rand_score(hidden_state, labels)}")
