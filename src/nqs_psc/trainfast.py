import netket as nk
from nqs_psc.utils import save_run

# Taille du système
L = 3

# Définition de l'hamiltonien

g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ham = nk.operator.Ising(hi, g, h=3, J=1)


# Création de l'état variationnel

model = nk.models.RBM()
sampler = nk.sampler.MetropolisLocal(hi, n_chains=300)
vstate = nk.vqs.MCState(sampler, model, n_samples=1000)

# Optimisation

optimizer = nk.optimizer.Sgd(learning_rate=0.01)
gs = nk.driver.VMC(ham, optimizer, variational_state=vstate)

# création du logger

log = nk.logging.RuntimeLog()

# One or more logger objects must be passed to the keyword argument `out`.
gs.run(n_iter=300, out=log)


meta = {
    "L": L,
    "graph": "Hypercube",
    "n_dim": 2,
    "pbc": True,
    "hamiltonian": {"type": "Ising", "h": 3, "J": 1},
    "model": "GCNN",
    "sampler": {"type": "MetropolisLocal", "n_chains": 300, "n_samples": 1000},
    "optimizer": {"type": "SGD", "lr": 0.01},
    "n_iter": 300,
}

vstate_dict = vstate.to_dict()
run_dir = save_run(log, meta, vstate_dict)
print("test")
