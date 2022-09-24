################################################
################## IMPORT ######################
################################################

import json
import sys
from datetime import datetime
from functools import partial, wraps

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax_md import space
from shadow.plot import *
from sklearn.metrics import r2_score

from psystems.npendulum import (PEF, edge_order, get_init, hconstraints,
                                pendulum_connections)

# from statistics import mode


# from sympy import LM


# from torch import batch_norm_gather_stats_with_counts


MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import fgn, lnn
from src.graph import *
from src.lnn import acceleration, accelerationFull, accelerationTV
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import nve
from src.utils import *

# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")


def wrap_main(f):
    def fn(*args, **kwargs):
        config = (args, kwargs)
        print("Configs: ")
        print(f"Args: ")
        for i in args:
            print(i)
        print(f"KwArgs: ")
        for k, v in kwargs.items():
            print(k, ":", v)
        return f(*args, **kwargs, config=config)

    return fn

N=5
epochs=10000
seed=42
rname=True
saveat=100
error_fn="L2error"
dt=1.0e-5
ifdrag=0
stride=1000
trainm=1
grid=False
mpass=1
lr=0.001
withdata=None
datapoints=None
batch_size=100
config=None

# def Main(N=5, epochs=10000, seed=42, rname=True, saveat=10, error_fn="L2error",
#          dt=1.0e-3, ifdrag=0, stride=100, trainm=1, grid=False, mpass=1, lr=0.001,
#          withdata=None, datapoints=None, batch_size=100):
    
#     return wrap_main(main)(N=N, epochs=epochs, seed=seed, rname=rname, saveat=saveat, error_fn=error_fn,
#                            dt=dt, ifdrag=ifdrag, stride=stride, trainm=trainm, grid=grid, mpass=mpass, lr=lr,
#                            withdata=withdata, datapoints=datapoints, batch_size=batch_size)


# def main(N=5, epochs=10000, seed=42, rname=True, saveat=10, error_fn="L2error",
#          dt=1.0e-3, ifdrag=0, stride=100, trainm=1, grid=False, mpass=1, lr=0.001, withdata=None, datapoints=None, batch_size=100, config=None):
    
    # print("Configs: ")
    # pprint(N, epochs, seed, rname,
    #        dt, stride, lr, ifdrag, batch_size,
    #        namespace=locals())

randfilename = datetime.now().strftime(
    "%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"

PSYS = f"ham-{N}-Pendulum"
TAG = f"6HGN"
out_dir = f"../results"

def _filename(name, tag=TAG):
    rstring = randfilename if (rname and (tag != "data")) else (
        "0" if (tag == "data") or (withdata == None) else f"{withdata}")
    filename_prefix = f"{out_dir}/{PSYS}-{tag}/{rstring}/"
    file = f"{filename_prefix}/{name}"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    filename = f"{filename_prefix}/{name}".replace("//", "/")
    print("===", filename, "===")
    return filename

def displacement(a, b):
    return a - b

def shift(R, dR, V):
    return R+dR, V

def OUT(f):
    @wraps(f)
    def func(file, *args, tag=TAG, **kwargs):
        return f(_filename(file, tag=tag), *args, **kwargs)
    return func

loadmodel = OUT(src.models.loadmodel)
savemodel = OUT(src.models.savemodel)

loadfile = OUT(src.io.loadfile)
savefile = OUT(src.io.savefile)
save_ovito = OUT(src.io.save_ovito)

savefile(f"config_{ifdrag}_{trainm}.pkl", config)

################################################
################## CONFIG ######################
################################################
np.random.seed(seed)
key = random.PRNGKey(seed)

try:
    dataset_states = loadfile(f"model_states_{ifdrag}.pkl", tag="data")[0]
except:
    raise Exception("Generate dataset first. Use *-data.py file.")

if datapoints is not None:
    dataset_states = dataset_states[:datapoints]

model_states = dataset_states[0]
z_out, zdot_out = model_states

print(
    f"Total number of data points: {len(dataset_states)}x{z_out.shape[0]}")



N2, dim = z_out.shape[-2:]
N = N2//2
species = jnp.zeros((N, 1), dtype=int)
masses = jnp.ones((N, 1))

array = jnp.array([jnp.array(i) for i in dataset_states])

Zs = array[:, 0, :, :, :]
Zs_dot = array[:, 1, :, :, :]

Zs = Zs.reshape(-1, N2, dim)
Zs_dot = Zs_dot.reshape(-1, N2, dim)

mask = np.random.choice(len(Zs), len(Zs), replace=False)
allZs = Zs[mask]
allZs_dot = Zs_dot[mask]

Ntr = int(0.75*len(Zs))
Nts = len(Zs) - Ntr

Zs = allZs[:Ntr]
Zs_dot = allZs_dot[:Ntr]

Zst = allZs[Ntr:]
Zst_dot = allZs_dot[Ntr:]


################################################
################## SYSTEM ######################
################################################
# def phi(x):
#     X = jnp.vstack([x[:1, :]*0, x])
#     return jnp.square(X[:-1, :] - X[1:, :]).sum(axis=1) - 1.0


# constraints = get_constraints(N, dim, phi)

################################################
################### ML Model ###################
################################################

senders, receivers = pendulum_connections(N)
eorder = edge_order(N)

hidden_dim = [16, 16]
edgesize = 1
nodesize = 5
ee = 8
ne = 8
Hparams = dict(
    ee_params=initialize_mlp([edgesize, ee], key),
    ne_params=initialize_mlp([nodesize, ne], key),
    e_params=initialize_mlp([ee+2*ne, *hidden_dim, ee], key),
    n_params=initialize_mlp([2*ee+ne, *hidden_dim, ne], key),
    g_params=initialize_mlp([ne, *hidden_dim, 1], key),
    acc_params=initialize_mlp([ne, *hidden_dim, 2*dim], key),
)

Z = Zs[0]
R, V = jnp.split(Zs[0], 2, axis=0)
species = jnp.array(species).reshape(-1, 1)

def dist(*args):
    disp = displacement(*args)
    return jnp.sqrt(jnp.square(disp).sum())

dij = vmap(dist, in_axes=(0, 0))(R[senders], R[receivers])


def acceleration_fn(params, graph):
    acc = fgn.cal_acceleration(params, graph, mpass=1)
    return acc

def acc_fn(species):
    state_graph = jraph.GraphsTuple(nodes={
        "position": R,
        "velocity": V,
        "type": species
    },
        edges={"dij": dij},
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([R.shape[0]]),
        n_edge=jnp.array([senders.shape[0]]),
        globals={})
    
    def apply(R, V, params):
        state_graph.nodes.update(position=R)
        state_graph.nodes.update(velocity=V)
        state_graph.edges.update(dij=vmap(dist, in_axes=(0, 0))(R[senders], R[receivers])
                                    )
        return acceleration_fn(params, state_graph)
    return apply

apply_fn = acc_fn(species)
v_apply_fn = vmap(apply_fn, in_axes=(None, 0))

def acceleration_fn_model(x, v, params): return apply_fn(x, v, params["H"])

params = {"H": Hparams}

print(acceleration_fn_model(R,V,params))

# print("lag: ", Lmodel(R, V, params))

# def nndrag(v, params):
#     return - jnp.abs(models.forward_pass(params, v.reshape(-1), activation_fn=models.SquarePlus)) * v

# if ifdrag == 0:
#     print("Drag: 0.0")

#     def drag(x, v, params):
#         return 0.0
# elif ifdrag == 1:
#     print("Drag: nn")

#     def drag(x, v, params):
#         return vmap(nndrag, in_axes=(0, None))(v.reshape(-1), params["drag"]).reshape(-1, 1)

# params["drag"] = initialize_mlp([1, 5, 5, 1], key)

# acceleration_fn_model = jit(accelerationFull(N, dim,
#                                              lagrangian=Lmodel,
#                                              constraints=None,
#                                              non_conservative_forces=drag))

v_acceleration_fn_model = vmap(acceleration_fn_model, in_axes=(0, 0, None))

################################################
################## ML Training #################
################################################

LOSS = getattr(src.models, error_fn)

print(acceleration_fn_model(R,V,params))

# pred = v_acceleration_fn_model(bRs[0], bVs[0], params)
# pred.shape
# zdot_pred = jnp.vstack(jnp.split(pred, 2, axis=2))

@jit
def loss_fn(params, Rs, Vs, Zs_dot):
    pred = v_acceleration_fn_model(Rs, Vs, params)
    zdot_pred = jnp.hstack(jnp.split(pred, 2, axis=2))
    return LOSS(zdot_pred, Zs_dot)

@jit
def gloss(*args):
    return value_and_grad(loss_fn)(*args)

opt_init, opt_update_, get_params = optimizers.adam(lr)

@ jit
def opt_update(i, grads_, opt_state):
    grads_ = jax.tree_map(jnp.nan_to_num, grads_)
    grads_ = jax.tree_map(
        partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
    return opt_update_(i, grads_, opt_state)

@jit
def update(i, opt_state, params, loss__, *data):
    """ Compute the gradient for a batch and update the parameters """
    value, grads_ = gloss(params, *data)
    opt_state = opt_update(i, grads_, opt_state)
    return opt_state, get_params(opt_state), value

@ jit
def step(i, ps, *args):
    return update(i, *ps, *args)

def batching(*args, size=None):
    L = len(args[0])
    if size != None:
        nbatches1 = int((L - 0.5) // size) + 1
        nbatches2 = max(1, nbatches1 - 1)
        size1 = int(L/nbatches1)
        size2 = int(L/nbatches2)
        if size1*nbatches1 > size2*nbatches2:
            size = size1
            nbatches = nbatches1
        else:
            size = size2
            nbatches = nbatches2
    else:
        nbatches = 1
        size = L
    
    newargs = []
    for arg in args:
        newargs += [jnp.array([arg[i*size:(i+1)*size]
                                for i in range(nbatches)])]
    return newargs

Rs, Vs = jnp.split(Zs, 2, axis=1)
Rst, Vst = jnp.split(Zst, 2, axis=1)
    
bRs, bVs, bZs_dot = batching(Rs, Vs, Zs_dot,
                            size=min(len(Rs), batch_size))
# bZs_dot.shape
# bRs.shape
print(f"training ...")

opt_state = opt_init(params)
epoch = 0
optimizer_step = -1
larray = []
ltarray = []
last_loss = 1000

larray += [loss_fn(params, Rs, Vs, Zs_dot)]
ltarray += [loss_fn(params, Rst, Vst, Zst_dot)]

def print_loss():
    print(
        f"Epoch: {epoch}/{epochs} Loss (mean of {error_fn}):  train={larray[-1]}, test={ltarray[-1]}")

print_loss()

for epoch in range(epochs):
    for data in zip(bRs, bVs, bZs_dot):
        optimizer_step += 1
        opt_state, params, l_ = step(
            optimizer_step, (opt_state, params, 0), *data)
    
    # optimizer_step += 1
    # opt_state, params, l_ = step(
    #     optimizer_step, (opt_state, params, 0), Rs, Vs, Fs)
    
    if epoch % saveat == 0:
        larray += [loss_fn(params, Rs, Vs, Zs_dot)]
        ltarray += [loss_fn(params, Rst, Vst, Zst_dot)]
        print_loss()
    
    if epoch % saveat == 0:
        metadata = {
            "savedat": epoch,
            "mpass": mpass,
            "grid": grid,
            "ifdrag": ifdrag,
            "trainm": trainm,
        }
        savefile(f"fgn_trained_model_{ifdrag}_{trainm}.dil",
                    params, metadata=metadata)
        savefile(f"loss_array_{ifdrag}_{trainm}.dil",
                    (larray, ltarray), metadata=metadata)
        if last_loss > larray[-1]:
            last_loss = larray[-1]
            savefile(f"fgn_trained_model_{ifdrag}_{trainm}_low.dil",
                        params, metadata=metadata)
        
        fig, axs = panel(1, 1)
        plt.semilogy(larray[1:], label="Training")
        plt.semilogy(ltarray[1:], label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))
        
fig, axs = panel(1, 1)
plt.semilogy(larray[1:], label="Training")
plt.semilogy(ltarray[1:], label="Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))

metadata = {
    "savedat": epoch,
    "mpass": mpass,
    "grid": grid,
    "ifdrag": ifdrag,
    "trainm": trainm,
}
params = get_params(opt_state)
savefile(f"fgn_trained_model_{ifdrag}_{trainm}.dil",
            params, metadata=metadata)
savefile(f"loss_array_{ifdrag}_{trainm}.dil",
            (larray, ltarray), metadata=metadata)


# fire.Fire(Main)
