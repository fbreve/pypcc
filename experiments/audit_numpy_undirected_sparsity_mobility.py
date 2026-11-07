"""Canonical *undirected* Digits k3 vs k10: full-iteration native NumPy mobility.

Frozen cohort in numpy_undirected_sparsity_mobility_protocol.md.
No changes to PCC algorithms, stopping, random numbers or graph construction.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_graph import build_knn_graph
from pcc_numpy import pcc_propagate_numpy
from experiments.numpy_continuous_observer import clone_rng, rng_equal
from experiments.audit_numpy_trajectory_dynamics import raw_state, identical_raw

SEEDS=(20107,20201,20309,20411)
KS=(3,10)
CAP=100000
WINDOW=1000
N_LABEL_PER_CLASS=2
PARAMS=dict(p_grd=.5,delta_v=.1,deltap=1.,dexp=2.)


def canonical_inputs(seed,k):
    data=load_digits()
    truth=np.asarray(data.target,dtype=np.int64)
    x=StandardScaler().fit_transform(np.asarray(data.data,dtype=np.float64))
    neighbors,degrees=build_knn_graph(x,k_nn=k)
    nodes=len(truth)
    if (neighbors.shape[0]!=nodes or degrees.shape!=(nodes,)
            or int(np.min(degrees))<=0):
        raise AssertionError("Invalid or isolated canonical production vertices")
    active_sets=[]
    rows=[]
    cols=[]
    for node,degree in enumerate(degrees):
        d=int(degree)
        a=[int(n) for n in neighbors[node,:d]]
        if (len(set(a))!=d or any(n==node or n<0 or n>=nodes for n in a)):
            raise AssertionError("Self-loop/duplicate/out-of-bounds active neighbor")
        active_sets.append(set(a))
        rows.extend([node]*d)
        cols.extend(a)
    if any(i not in active_sets[j]
           for i,adj in enumerate(active_sets) for j in adj):
        raise AssertionError("Noncanonical directed edge in official union kNN")
    adj=csr_matrix((np.ones(len(rows),dtype=np.int8),(rows,cols)),
                   shape=(nodes,nodes))
    ncomponents,components=connected_components(adj,directed=False)
    component_sizes=np.bincount(components,minlength=ncomponents)
    labels=np.full(nodes,-1,dtype=np.int64)
    rng=np.random.default_rng(seed)
    for cls in np.unique(truth):
        choices=rng.choice(np.flatnonzero(truth==cls),
                           N_LABEL_PER_CLASS,replace=False)
        labels[choices]=int(cls)
    if np.any(np.bincount(labels[labels!=-1],minlength=10)!=2):
        raise AssertionError("Wrong frozen particle/class count")
    return neighbors,degrees,labels,truth,dict(
        requested_k=k,n_nodes=nodes,
        mean_degree=float(np.mean(degrees)),
        min_degree=int(np.min(degrees)),max_degree=int(np.max(degrees)),
        ncomponents=int(ncomponents),
        component_sizes=sorted(map(int,component_sizes),reverse=True),
        directed_edges=len(rows),reciprocal=True,no_self_edges=True)


class MobilityObserver:
    """Streaming read-only per-step motion + every-window class metrics."""
    def __init__(self,mapped,graph,degrees,*,cap=CAP,window=WINDOW):
        if cap<1 or window<1 or cap%window:
            raise ValueError("cap must be positive and divisible by window")
        self.mapped=np.asarray(mapped,dtype=np.int64).copy()
        self.graph=graph
        self.degrees=degrees
        self.cap=cap
        self.window=window
        self.classes=int(np.max(self.mapped))+1
        self.particle_classes=self.mapped[self.mapped!=-1].copy()
        self.nparticles=len(self.particle_classes)
        self.previous=None
        self.streak=np.zeros(self.nparticles,dtype=np.int64)
        self.max_streak=np.zeros_like(self.streak)
        self.ever_moved=np.zeros(self.nparticles,dtype=bool)
        self.window_moves=np.zeros(self.classes,dtype=np.int64)
        self.total_moves_by_class=np.zeros(self.classes,dtype=np.int64)
        self.zero_movement_steps=0
        self.zero_class_streak=np.zeros(self.classes,dtype=np.int64)
        self.max_zero_class_streak=np.zeros(self.classes,dtype=np.int64)
        self.first_operational_event=[None]*self.classes
        self.first_recovery=[None]*self.classes
        self.windows=[]
        self.last_iteration=-1
        self.real_callbacks=0
        self.final=None

    def wants(self,iteration):
        return True

    def capture(self,iteration,dominance,positions,strengths,distances,
                ownership,rng,info):
        if info is not None:
            if (self.last_iteration!=self.cap or iteration!=self.cap
                    or self.final is not None):
                raise AssertionError("Invalid final notification")
            self.final=dict(iteration=iteration,
                info=dict(info),rng=clone_rng(rng),
                raw=raw_state(dominance,positions,strengths,distances,
                              ownership,self.mapped))
            return
        if self.final is not None:
            raise AssertionError("Row sent after true finalization")
        if iteration!=self.last_iteration+1 or not 0<=iteration<=self.cap:
            raise AssertionError("Duplicate or missed physical iteration")
        self.last_iteration=iteration
        self.real_callbacks+=1
        curr=np.asarray(positions)
        if curr.ndim!=1 or curr.shape[0]!=self.nparticles:
            raise AssertionError("Invalid live particle positions")
        if iteration==0:
            self.previous=curr.copy()
            return

        moved=(curr!=self.previous)
        nmove=int(np.count_nonzero(moved))
        if nmove==0:
            self.zero_movement_steps+=1
        perclass=np.bincount(self.particle_classes[moved],
                            minlength=self.classes)
        self.window_moves+=perclass
        self.total_moves_by_class+=perclass
        self.streak=np.where(moved,0,self.streak+1)
        self.max_streak=np.maximum(self.streak,self.max_streak)
        self.ever_moved|=moved
        self.previous=curr.copy()

        if iteration%self.window:
            return

        if (not np.all(np.isfinite(dominance))
                or not np.allclose(dominance.sum(axis=1),1.,
                                   rtol=0,atol=1e-8)):
            raise AssertionError("Nonfinite or non-simplex dominance")
        strengths=np.asarray(strengths)
        if not np.all(np.isfinite(strengths)):
            raise AssertionError("Nonfinite particle strength")
        unknown=self.mapped==-1
        mass=np.asarray(dominance)[unknown].sum(axis=0)
        for cls in range(self.classes):
            ids=np.flatnonzero(self.particle_classes==cls)
            if len(ids)!=N_LABEL_PER_CLASS:
                raise AssertionError("Wrong per-class initial particles")
            places=curr[ids]
            if np.any(places<0) or np.any(places>=len(self.mapped)):
                raise AssertionError("Particle escaped graph")
            if np.any(self.degrees[places]<=0):
                raise AssertionError("Invalid/isolated destination")
            # All active vertices have valid non-self neighbors. One visit
            # per particle per physical native iteration is guaranteed.
            accepted=int(self.window_moves[cls])
            attempts=self.window*len(ids)
            if not 0<=accepted<=attempts:
                raise AssertionError("Impossible accepted-movement count")
            self.zero_class_streak[cls]=(
                self.zero_class_streak[cls]+1 if accepted==0 else 0)
            self.max_zero_class_streak[cls]=max(
                int(self.max_zero_class_streak[cls]),
                int(self.zero_class_streak[cls]))
            winning=0
            nentries=0
            for vertex in places:
                n=int(self.degrees[vertex])
                outgoing=self.graph[vertex,:n]
                winning+=int(np.count_nonzero(
                    dominance[outgoing,cls]==np.max(dominance[outgoing],axis=1)))
                nentries+=n
            current_strength=strengths[ids]
            minimum=float(np.min(current_strength))
            maximum=float(np.max(current_strength))
            if minimum < -1e-8 or maximum >1.+1e-8:
                raise AssertionError("Particle strength outside simplex bounds")
            locked=(self.zero_class_streak[cls]>=5
                    and maximum<=.01 and winning==0)
            if locked and self.first_operational_event[cls] is None:
                self.first_operational_event[cls]=iteration
            if (self.first_operational_event[cls] is not None
                    and self.first_recovery[cls] is None and accepted>0):
                self.first_recovery[cls]=iteration
            self.windows.append(dict(
                start=iteration-self.window,end=iteration,class_id=cls,
                attempts=attempts,accepted_moves=accepted,
                acceptance_fraction=float(accepted/attempts),
                zero_move_streak=int(self.zero_class_streak[cls]),
                operational_lockout=bool(locked),
                max_strength=maximum,min_strength=minimum,
                winning_outgoing_entries=winning,
                neighbor_entries=nentries,
                dominance_mass=float(mass[cls])))
        self.window_moves[:]=0

    def summary(self):
        if (self.real_callbacks!=self.cap+1 or
                self.last_iteration!=self.cap or self.final is None or
                len(self.windows)!=(self.cap//self.window)*self.classes):
            raise AssertionError("Incomplete streaming trajectory/summary")
        if self.final["info"]["effective_iterations"]!=self.cap:
            raise AssertionError("No-stop pilot unexpectedly stopped")
        affected=sum(v is not None for v in self.first_operational_event)
        return dict(
            physical_iterations=self.real_callbacks,
            n_windows=self.cap//self.window,
            total_moves_by_class=self.total_moves_by_class.tolist(),
            total_accepted_moves=int(np.sum(self.total_moves_by_class)),
            zero_movement_steps=self.zero_movement_steps,
            individual_max_streaks=self.max_streak.tolist(),
            longest_individual_streak=int(self.max_streak.max())
                if self.nparticles else 0,
            never_moved_particles=int(np.count_nonzero(~self.ever_moved)),
            class_max_zero_window_streaks=self.max_zero_class_streak.tolist(),
            first_class_lockout=self.first_operational_event,
            first_recovered_move=self.first_recovery,
            operationally_locked_classes=affected,
            any_operational_lockout=bool(affected),
            zero_move_class_windows=sum(
                int(row["accepted_moves"]==0) for row in self.windows),
            windows=self.windows)


class MeasuredPCC(PCC):
    def __init__(self,record):
        super().__init__(impl="numpy")
        self.record=record
        self.audit=None
        self.info=None

    def _get_backend_fn(self):
        native=PCC._get_backend_fn(self)
        if native is not pcc_propagate_numpy:
            raise AssertionError("Expected native NumPy propagation")
        def call(*args,**kwargs):
            if "observer" in kwargs or "return_info" in kwargs:
                raise AssertionError("Unexpected public observer forwarding")
            if self.record:
                self.audit=MobilityObserver(
                    args[2],args[0],args[1])
                kwargs["observer"]=self.audit
            self.info=dict(native(*args,return_info=True,**kwargs))
            return None
        call.__name__=native.__name__
        return call


def run_fit(graph,degrees,labels,seed,k,record):
    model=MeasuredPCC(record)
    model.set_graph(graph,degrees,k_nn=k)
    start=time.perf_counter()
    predictions=model.fit_predict(
        labels.copy(),seed=seed,max_iter=CAP,early_stop=False,
        es_chk=2000,**PARAMS)
    elapsed=time.perf_counter()-start
    state=dict(
        predictions=np.asarray(predictions).copy(),
        positions=model.part.curnode.copy(),
        strengths=model.part.strength.copy(),
        distances=model.part.dist_table.copy(),
        dominance=model.node.dominance.copy(),
        ownership=model.owndeg.copy())
    rng=clone_rng(np.random.get_state())
    following=np.random.random(16)
    if not np.array_equal(predictions[labels!=-1],labels[labels!=-1]):
        raise AssertionError("Labeled vertices changed")
    return dict(model=model,state=state,info=model.info,rng=rng,
                following=following,seconds=elapsed)


def verify_actual_final(result):
    obs=result["model"].audit
    if obs is None or obs.final is None:
        raise AssertionError("Missing streaming final notification")
    last=obs.final
    state=result["state"]
    raw=last["raw"]
    mapped=result["model"].mapped_labels
    assignments=np.argmax(state["dominance"],axis=1)
    assignments[mapped!=-1]=mapped[mapped!=-1]
    sums=raw["raw_ownership"].sum(axis=1,keepdims=True)
    sums[sums==0]=1.
    checks=(
        np.array_equal(raw["assignments"],assignments),
        np.array_equal(raw["positions"],state["positions"]),
        np.array_equal(raw["strengths"],state["strengths"]),
        np.array_equal(raw["distances"],state["distances"]),
        np.array_equal(raw["dominance"],state["dominance"]),
        np.array_equal(raw["raw_ownership"]/sums,state["ownership"]),
        rng_equal(last["rng"],result["rng"]),
        last["info"]==result["info"],last["iteration"]==CAP)
    if not all(checks):
        raise AssertionError(f"Native final snapshot/RNG mismatch: {checks}")


def evaluate(seed,outfile):
    result=dict(seed=seed,cap=CAP,window=WINDOW,classes=10,
                dataset="digits",graph_type="canonical_undirected_union_knn",
                cases=[])
    for k in KS:
        graph,degrees,labels,truth,structure=canonical_inputs(seed,k)
        order=(False,True) if (SEEDS.index(seed)+KS.index(k))%2==0 else (True,False)
        runs={v:run_fit(graph,degrees,labels,seed,k,v) for v in order}
        a,b=runs[False],runs[True]
        states=all(np.array_equal(a["state"][key],b["state"][key])
                   for key in a["state"])
        info=a["info"]==b["info"]
        rng_equal_final=rng_equal(a["rng"],b["rng"])
        following_equal=np.array_equal(a["following"],b["following"])
        if not all((states,info,rng_equal_final,following_equal)):
            raise AssertionError(f"Observed/unobserved bitwise mismatch seed={seed},k={k}")
        verify_actual_final(b)
        dynamics=b["model"].audit.summary()
        unknown=labels==-1
        acc=float(np.mean(a["state"]["predictions"][unknown]==truth[unknown]))
        result["cases"].append(dict(
            k=k,structure=structure,labels_count=int(np.sum(~unknown)),
            label_positions=np.flatnonzero(~unknown).tolist(),
            accuracy_unlabeled=acc,
            full_six_state_exact=states,
            full_native_stop_exact=info,
            full_mt19937_rng_exact=rng_equal_final,
            next_16_draws_exact=following_equal,
            final_low_level_state_exact=True,
            times_seconds=dict(baseline=a["seconds"],
                               observed=b["seconds"]),
            dynamics=dynamics))
        Path(outfile).write_text(json.dumps(
            result,indent=2,sort_keys=True,allow_nan=False)+"\n",
            encoding="utf-8")
        print(f"Digits seed {seed}, canonical union k={k}: "
              f"{dynamics['operationally_locked_classes']} class lockouts, "
              f"{dynamics['zero_move_class_windows']} zero-move class-windows, "
              f"max particle still {dynamics['longest_individual_streak']} steps, "
              f"accuracy={acc:.4f}; full state/RNG parity PASS",
              flush=True)
    if result["cases"][0]["label_positions"]!=result["cases"][1]["label_positions"]:
        raise AssertionError("Sparse/dense paired label subsets differ")
    result["paired_accuracy_sparse_minus_dense_pp"]=100*(
        result["cases"][0]["accuracy_unlabeled"]-
        result["cases"][1]["accuracy_unlabeled"])
    Path(outfile).write_text(json.dumps(
        result,indent=2,sort_keys=True,allow_nan=False)+"\n",
        encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True,choices=SEEDS)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    evaluate(a.seed,a.output)


if __name__=="__main__":
    main()
