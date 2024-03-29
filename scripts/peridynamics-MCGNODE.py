################################################
################## IMPORT ######################
################################################

import pandas as pd
import json
import sys
from datetime import datetime
from functools import partial, wraps
from statistics import mode

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax_md import space
from shadow.plot import *
from sklearn.metrics import r2_score
# from sympy import LM
# from torch import batch_norm_gather_stats_with_counts

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import lnn
from src.graph import *
from src.lnn import acceleration, accelerationFull, accelerationTV, acceleration_GNODE
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import nve
from src.utils import *


config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")


# import pickle
# data = pickle.load(open('../results/LJ-data/0/graphs_dicts.pkl','rb'))[0]
# dd = data[0]['nodes']['position']
# data[1]


acceleration = []
damage = []
id = []
mass = []
position = []
type = []
velocity = []
volume = []
for num in (np.linspace(0, 5000, 251).astype('int')):
    dataf_name = f"env_1_step_{num}.jld.data"
    df = pd.read_csv(f'../results/peridynamics-data/datafiles/{dataf_name}')
    split_df = df.iloc[1:, 0].str.split(expand=True)
    acceleration += [(np.array(split_df[[0, 1, 2]]).astype('float64'))]
    damage += [np.array(split_df[[3]]).astype('float64')]
    id += [np.array(split_df[[4]]).astype('float64')]
    mass += [np.array(split_df[[5]]).astype('float64')]
    position += [np.array(split_df[[6, 7, 8]]).astype('float64')]
    type += [np.array(split_df[[9]]).astype('float64')]
    velocity += [np.array(split_df[[10, 11, 12]]).astype('float64')]
    volume += [np.array(split_df[[13]]).astype('float64')]


Rs = jnp.array(position)
Vs = jnp.array(velocity)
Fs = jnp.array(acceleration)


o_position = position[0]/1.1
N, dim = o_position.shape
species = jnp.zeros(N, dtype=int)


def displacement(a, b):
    return a - b


# make_graph(o_position,displacement[0],species=species,atoms={0: 125},V=velocity[0],A=acceleration[0],mass=mass[0],cutoff=3.0)
my_graph0_disc = make_graph(
    o_position, displacement, atoms={0: 125}, cutoff=3.0)


epochs = 10000
seed = 42
rname = True
dt = 1.0e-3
ifdrag = 0
stride = 100
trainm = 1
lr = 0.001
withdata = None
datapoints = None
batch_size = 20


# def main(N=5, epochs=10000, seed=42, rname=True, dt=1.0e-3, ifdrag=0, stride=100, trainm=1, lr=0.001, withdata=None, datapoints=None, batch_size=100):
# print("Configs: ")
# pprint(N, epochs, seed, rname,
#         dt, stride, lr, ifdrag, batch_size,
#         namespace=locals())

randfilename = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"

PSYS = f"peridynamics"
TAG = f"MC-MCGNODE"
out_dir = f"../results"


def _filename(name, tag=TAG):
    rstring = randfilename if (rname and (tag != "data")) else (
        "0" if (tag == "data") or (withdata == None) else f"0_{withdata}")
    filename_prefix = f"{out_dir}/{PSYS}-{tag}/{rstring}/"
    file = f"{filename_prefix}/{name}"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    filename = f"{filename_prefix}/{name}".replace("//", "/")
    print("===", filename, "===")
    return filename

# def displacement(a, b):
#     return a - b


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

################################################
################## CONFIG ######################
################################################
np.random.seed(seed)
key = random.PRNGKey(seed)

# try:
#     graphs = loadfile(f"env_1_step_0.jld.data", tag="data")
# except:
#     raise Exception("Generate dataset first.")


species = jnp.zeros(N, dtype=int)
masses = jnp.ones(N)

# Rs, Vs, Fs = States(graphs).get_array()


mask = np.random.choice(len(Rs), len(Rs), replace=False)
allRs = Rs[mask]
allVs = Vs[mask]
allFs = Fs[mask]


Ntr = int(0.75*len(allRs))
Nts = len(allRs) - Ntr

Rs = allRs[:Ntr]
Vs = allVs[:Ntr]
Fs = allFs[:Ntr]

Rst = allRs[Ntr:]
Vst = allVs[Ntr:]
Fst = allFs[Ntr:]

print(f"training data shape(Rs): {Rs.shape}")
print(f"test data shape(Rst): {Rst.shape}")

################################################
################## SYSTEM ######################
################################################

# peridynamics_sim

################################################
################### ML Model ###################
################################################

dim = 3
Ef = dim  # eij dim
Nf = dim
Oh = 1

Eei = 8
Nei = 8
Nei_ = 5  # Nei for mass

hidden = 8
nhidden = 2


def get_layers(in_, out_):
    return [in_] + [hidden]*nhidden + [out_]


def mlp(in_, out_, key, **kwargs):
    return initialize_mlp(get_layers(in_, out_), key, **kwargs)


fneke_params = initialize_mlp([Oh, Nei], key)
fne_params = initialize_mlp([Oh, Nei], key)  #

# Nei = Nei+dim+dim
fb_params = mlp(Ef, Eei, key)  #
fv_params = mlp(Nei+Eei, Nei, key)  #
fe_params = mlp(Nei, Eei, key)  #

ff1_params = mlp(Eei, dim, key)
ff2_params = mlp(Nei, dim, key)
ff3_params = mlp(Nei+dim+dim, dim, key)
ke_params = initialize_mlp([1+Nei, 10, 10, 1], key, affine=[True])
mass_params = initialize_mlp([Nei_, 5, 1], key, affine=[True])

Fparams = dict(fb=fb_params,
               fv=fv_params,
               fe=fe_params,
               ff1=ff1_params,
               ff2=ff2_params,
               ff3=ff3_params,
               fne=fne_params,
               fneke=fneke_params,
               ke=ke_params,
               mass=mass_params)

params = {"Fqqdot": Fparams}


def graph_force_fn(params, graph):
    _GForce = a_mc_mcgnode_cal_force_q_qdot(params, graph, eorder=None,
                                            useT=True)
    return _GForce


R, V = Rs[0], Vs[0]

my_graph0_disc.pop("e_order")
my_graph0_disc.pop("atoms")
my_graph0_disc.update({"globals": None})

mask = my_graph0_disc['senders'] != my_graph0_disc['receivers']
my_graph0_disc.update({"senders": my_graph0_disc['senders'][mask]})
my_graph0_disc.update({"receivers": my_graph0_disc['receivers'][mask]})
my_graph0_disc.update({"n_edge": mask.sum()})

graph = jraph.GraphsTuple(**my_graph0_disc)


def _force_fn(species):
    state_graph = graph

    def apply(R, V, params):
        state_graph.nodes.update(position=R)
        state_graph.nodes.update(velocity=V)
        return graph_force_fn(params, state_graph)
    return apply


apply_fn = _force_fn(species)
# v_apply_fn = vmap(apply_fn, in_axes=(None, 0))
apply_fn(R, V, Fparams)


def F_q_qdot(x, v, params): return apply_fn(x, v, params["Fqqdot"])


N, dim = R.shape
acceleration_fn_model = F_q_qdot
# acceleration_fn_model = acceleration_GNODE(N, dim, F_q_qdot,
#                                             constraints=None)


v_acceleration_fn_model = vmap(acceleration_fn_model, in_axes=(0, 0, None))

# v_acceleration_fn_model(Rs[:10], Vs[:10], params)


################################################
################## ML Training #################
################################################
@jit
def loss_fn(params, Rs, Vs, Fs):
    pred = v_acceleration_fn_model(Rs, Vs, params)
    return MSE(pred, Fs)

# loss_fn(params, Rs[:1], Vs[:1], Fs[:1])


def gloss(*args):
    return value_and_grad(loss_fn)(*args)


def update(i, opt_state, params, loss__, *data):
    """ Compute the gradient for a batch and update the parameters """
    value, grads_ = gloss(params, *data)
    opt_state = opt_update(i, grads_, opt_state)
    return opt_state, get_params(opt_state), value


@jit
def step(i, ps, *args):
    return update(i, *ps, *args)


opt_init, opt_update_, get_params = optimizers.adam(lr)


@jit
def opt_update(i, grads_, opt_state):
    grads_ = jax.tree_map(jnp.nan_to_num, grads_)
    grads_ = jax.tree_map(
        partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
    return opt_update_(i, grads_, opt_state)


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


bRs, bVs, bFs = batching(Rs, Vs, Fs,
                         size=min(len(Rs), batch_size))

print(f"training ...")

opt_state = opt_init(params)
epoch = 0
optimizer_step = -1
larray = []
ltarray = []
last_loss = 1000
for epoch in range(epochs):
    l = 0.0
    for data in zip(bRs, bVs, bFs):
        optimizer_step += 1
        opt_state, params, l_ = step(
            optimizer_step, (opt_state, params, 0), *data)
        l += l_

    opt_state, params, l_ = step(
        optimizer_step, (opt_state, params, 0), Rs, Vs, Fs)
    larray += [l_]
    ltarray += [loss_fn(params, Rst, Vst, Fst)]
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch}/{epochs} Loss (MSE):  train={larray[-1]}, test={ltarray[-1]}")
    if epoch % 10 == 0:
        metadata = {
            "savedat": epoch,
            # "mpass": mpass,
            "ifdrag": ifdrag,
            "trainm": trainm,
        }
        savefile(f"perimcgnode_trained_model_{ifdrag}_{trainm}.dil",
                 params, metadata=metadata)
        savefile(f"loss_array_{ifdrag}_{trainm}.dil",
                 (larray, ltarray), metadata=metadata)
        if last_loss > larray[-1]:
            last_loss = larray[-1]
            savefile(f"perimcgnode_trained_model_{ifdrag}_{trainm}_low.dil",
                     params, metadata=metadata)
        fig, axs = panel(1, 1)
        plt.semilogy(larray, label="Training")
        plt.semilogy(ltarray, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))

fig, axs = panel(1, 1)
plt.semilogy(larray, label="Training")
plt.semilogy(ltarray, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))

params = get_params(opt_state)
savefile(f"perimcgnode_trained_model_{ifdrag}_{trainm}.dil",
         params, metadata=metadata)
savefile(f"loss_array_{ifdrag}_{trainm}.dil",
         (larray, ltarray), metadata=metadata)

if last_loss > larray[-1]:
    last_loss = larray[-1]
    savefile(f"perimcgnode_trained_model_{ifdrag}_{trainm}_low.dil",
             params, metadata=metadata)

# fire.Fire(main)
