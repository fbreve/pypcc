"""Scientific pilot: every-iteration real NumPy PCC trajectories, with strict parity.

One native 6001-step fit is observed without restarting/prefix replay.
The observer streams bounded counters, 60 fixed windows and one stop checkpoint.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.audit_numpy_continuous_observer import graph_and_labels
from experiments.numpy_continuous_observer import clone_rng, rng_equal
from pcc import ParticleCompetitionAndCooperation as PCC
from pcc_numpy import pcc_propagate_numpy

SEEDS=(18013,18109)
MAX_ITER=6001
WINDOW=100
ARGUMENTS=dict(p_grd=.5,delta_v=.1,deltap=1.,dexp=2.)
STATE_KEYS=("assignments","positions","strengths","distances",
            "dominance","raw_ownership")


def raw_state(dominance,positions,strengths,distances,ownership,labels):
    assignments=np.argmax(dominance,axis=1).astype(np.int64)
    known=labels!=-1
    assignments[known]=labels[known]
    return dict(assignments=assignments,
                positions=np.array(positions,copy=True),
                strengths=np.array(strengths,copy=True),
                distances=np.array(distances,copy=True),
                dominance=np.array(dominance,copy=True),
                raw_ownership=np.array(ownership,copy=True))


def identical_raw(a,b):
    return all(np.array_equal(a[k],b[k]) for k in STATE_KEYS)


class FinalOnlyObserver:
    def __init__(self,mapped):
        self.mapped=np.asarray(mapped,dtype=np.int64).copy()
        self.final=None

    def wants(self,iteration):
        return False

    def capture(self,iteration,dominance,positions,strengths,distances,
                raw_ownership,rng,info):
        if info is None or self.final is not None:
            raise AssertionError("Final-only observer received unexpected callback")
        self.final=dict(iteration=iteration,
                        state=raw_state(dominance,positions,strengths,
                                        distances,raw_ownership,self.mapped),
                        rng=clone_rng(rng),info=dict(info))


class StreamingDynamicsObserver:
    """O(n*c + n_particles + 60 windows) memory; no RNG draws or writes."""
    def __init__(self,mapped,stop_iteration,stop_max):
        self.mapped=np.asarray(mapped,dtype=np.int64).copy()
        self.stop_iteration=int(stop_iteration)
        self.stop_max=int(stop_max)
        self.callbacks=0
        self.last_iteration=-1
        self.previous_assignments=None
        self.previous_positions=None
        self.window_assignments=None
        self.window_excursion=None
        self.window_dominance=None
        self.windows=[]
        self.step_flip_events=0
        self.step_movement_events=0
        self.zero_motion_steps=0
        self.streak=None
        self.max_streak=None
        self.ever_moved=None
        self.shadow_max=0.0
        self.shadow_stalled=0
        self.shadow_checks=0
        self.shadow_stop=None
        self.stop_checkpoint=None
        self.final_info=None

    def wants(self,iteration):
        return True

    def capture(self,iteration,dominance,positions,strengths,distances,
                raw_ownership,rng,info):
        if info is not None:
            if (iteration!=MAX_ITER or self.last_iteration!=MAX_ITER
                    or self.final_info is not None):
                raise AssertionError("Bad final notification or duplicate physical state")
            self.final_info=dict(info)
            return

        if iteration!=self.last_iteration+1 or not 0<=iteration<=MAX_ITER:
            raise AssertionError("Missing, out-of-order, or repeated native iteration")
        self.last_iteration=iteration
        self.callbacks+=1
        current=np.argmax(dominance,axis=1).astype(np.int64)
        known=self.mapped!=-1
        current[known]=self.mapped[known]
        if iteration==0:
            self.previous_assignments=current.copy()
            self.previous_positions=np.asarray(positions).copy()
            self.window_assignments=current.copy()
            self.window_excursion=np.zeros(current.shape,dtype=bool)
            self.window_dominance=np.asarray(dominance).copy()
            self.streak=np.zeros(len(positions),dtype=np.int64)
            self.max_streak=self.streak.copy()
            self.ever_moved=np.zeros(len(positions),dtype=bool)
        else:
            self.step_flip_events+=int(np.count_nonzero(
                current!=self.previous_assignments))
            moved=(np.asarray(positions)!=self.previous_positions)
            count=int(np.count_nonzero(moved))
            self.step_movement_events+=count
            if not count:
                self.zero_motion_steps+=1
            self.streak=np.where(moved,0,self.streak+1)
            self.max_streak=np.maximum(self.max_streak,self.streak)
            self.ever_moved|=moved

            self.window_excursion|=(current!=self.window_assignments)
            if iteration%WINDOW==0:
                net=(current!=self.window_assignments)
                hidden=self.window_excursion & ~net
                delta=np.max(np.abs(
                    np.asarray(dominance)-self.window_dominance))
                self.windows.append(dict(
                    start=iteration-WINDOW,end=iteration,
                    endpoint_changed_nodes=int(np.count_nonzero(net)),
                    reverted_nodes=int(np.count_nonzero(hidden)),
                    full_endpoint_stability=bool(not np.any(net)),
                    any_internal_excursion=bool(np.any(hidden)),
                    max_abs_endpoint_dominance_change=float(delta)))
                self.window_assignments=current.copy()
                self.window_dominance=np.asarray(dominance).copy()
                self.window_excursion[:]=False
            self.previous_assignments=current.copy()
            self.previous_positions=np.asarray(positions).copy()

            # This is the original NumPy propagation cadence: it%10 == 0,
            # where 'it' is zero-indexed and 'iteration' is one-indexed.
            if (iteration%10==1 and self.shadow_stop is None):
                potential=float(np.mean(np.max(dominance,axis=1)))
                self.shadow_checks+=1
                if potential>self.shadow_max:
                    self.shadow_max=potential
                    self.shadow_stalled=0
                else:
                    self.shadow_stalled+=1
                    if self.shadow_stalled>self.stop_max:
                        self.shadow_stop=iteration

        if iteration==self.stop_iteration:
            if self.stop_checkpoint is not None:
                raise AssertionError("Stop checkpoint recorded twice")
            self.stop_checkpoint=dict(
                iteration=iteration,
                state=raw_state(dominance,positions,strengths,distances,
                                raw_ownership,self.mapped),
                rng=clone_rng(rng))

    def summary(self):
        if (self.callbacks!=MAX_ITER+1 or self.last_iteration!=MAX_ITER
                or len(self.windows)!=MAX_ITER//WINDOW
                or self.stop_checkpoint is None
                or self.final_info is None):
            raise AssertionError("Incomplete single-run streamed dynamics")
        if (self.final_info["effective_iterations"]!=MAX_ITER
                or self.final_info["stop_reason"]!="max_iter"):
            raise AssertionError("Uninterrupted experiment unexpectedly stopped")
        quiet=[w for w in self.windows if w["full_endpoint_stability"]]
        return dict(
            native_callbacks=self.callbacks,
            full_windows=len(self.windows),
            step_flip_events=self.step_flip_events,
            step_movement_events=self.step_movement_events,
            zero_motion_steps=self.zero_motion_steps,
            quiet_endpoint_windows=len(quiet),
            quiet_but_hidden_windows=sum(
                w["any_internal_excursion"] for w in quiet),
            reverted_node_window_events=sum(w["reverted_nodes"]
                                            for w in self.windows),
            particles_never_changed_position=int(np.count_nonzero(
                ~self.ever_moved)),
            particles_final_stationary_ge_100=int(np.count_nonzero(
                self.streak>=100)),
            longest_stationary_steps=int(self.max_streak.max())
            if self.max_streak.size else 0,
            particle_max_streaks=self.max_streak.tolist(),
            shadow_first_stop_iteration=self.shadow_stop,
            shadow_checks_to_first_stop=self.shadow_checks,
            windows=self.windows)


class ScientificPCC(PCC):
    def __init__(self,kind,stop_iteration=None,stop_max=None):
        super().__init__(impl="numpy")
        self.kind=kind
        self.stop_iteration=stop_iteration
        self.stop_max=stop_max
        self.audit=None
        self.native_info=None

    def _get_backend_fn(self):
        native=PCC._get_backend_fn(self)
        if native is not pcc_propagate_numpy:
            raise AssertionError("Wrong backend in scientific NumPy pilot")

        def invoke(*args,**kwargs):
            if "observer" in kwargs or "return_info" in kwargs:
                raise AssertionError("fit_predict changed its private model calls")
            if self.kind=="stop":
                self.audit=FinalOnlyObserver(args[2])
                kwargs["observer"]=self.audit
            elif self.kind=="stream":
                self.audit=StreamingDynamicsObserver(
                    args[2],self.stop_iteration,self.stop_max)
                kwargs["observer"]=self.audit
            elif self.kind!="baseline":
                raise AssertionError("Unknown audit type")
            self.native_info=dict(native(*args,return_info=True,**kwargs))
            return None

        invoke.__name__=native.__name__
        return invoke


def run_fit(graph,degrees,labels,seed,kind,*,stopped=None):
    stop_iter=None if stopped is None else stopped["info"]["effective_iterations"]
    stop_max=None if stopped is None else stopped["info"]["stop_max"]
    model=ScientificPCC(kind,stop_iter,stop_max)
    model.set_graph(graph,degrees,k_nn=10)
    start=time.perf_counter()
    prediction=model.fit_predict(
        labels.copy(),seed=seed,max_iter=MAX_ITER,
        early_stop=(kind=="stop"),
        es_chk=10,**ARGUMENTS)
    elapsed=time.perf_counter()-start
    state=dict(predictions=np.asarray(prediction).copy(),
               positions=model.part.curnode.copy(),
               strengths=model.part.strength.copy(),
               distances=model.part.dist_table.copy(),
               dominance=model.node.dominance.copy(),
               ownership=model.owndeg.copy())
    if not np.array_equal(prediction[labels!=-1],labels[labels!=-1]):
        raise AssertionError("Original labeled nodes changed")
    rng_after=clone_rng(np.random.get_state())
    next16=np.random.random(16)
    return dict(model=model,state=state,info=model.native_info,
                rng=rng_after,next16=next16,
                seconds=elapsed)


def check_native_last(full):
    audit=full["model"].audit
    if not isinstance(audit,FinalOnlyObserver) or audit.final is None:
        raise AssertionError("No genuine terminal observer state")
    row=audit.final
    if (row["iteration"]!=full["info"]["effective_iterations"]
            or row["info"]!=full["info"] or not rng_equal(row["rng"],full["rng"])):
        raise AssertionError("Stop terminal state/info/RNG mismatch")
    state=row["state"]
    dest=full["state"]
    mapped=full["model"].mapped_labels
    raw_own=state["raw_ownership"]
    sums=raw_own.sum(axis=1,keepdims=True)
    sums[sums==0]=1.0
    predictions=np.argmax(dest["dominance"],axis=1)
    predictions[mapped!=-1]=mapped[mapped!=-1]
    # These four real datasets use labels 0..c-1, as also mapped internally.
    if not all((
        np.array_equal(state["positions"],dest["positions"]),
        np.array_equal(state["strengths"],dest["strengths"]),
        np.array_equal(state["distances"],dest["distances"]),
        np.array_equal(state["dominance"],dest["dominance"]),
        np.array_equal(raw_own/sums,dest["ownership"]),
        np.array_equal(state["assignments"],predictions),
        np.array_equal(predictions,dest["predictions"]),
    )):
        raise AssertionError("Final raw snapshot differs from high-level stop fit")
    return row


def evaluate_dataset(dataset,output):
    report={"dataset":dataset,"seeds":list(SEEDS),"cap":MAX_ITER,
            "window":WINDOW,"cases":[]}
    for seed in SEEDS:
        graph,degrees,labels,truth=graph_and_labels(dataset,seed)
        baseline=run_fit(graph,degrees,labels,seed,"baseline")
        stopped=run_fit(graph,degrees,labels,seed,"stop")
        terminal=check_native_last(stopped)
        scanned=run_fit(graph,degrees,labels,seed,"stream",stopped=stopped)
        scan=scanned["model"].audit
        dynamics=scan.summary()

        six=all(np.array_equal(baseline["state"][k],scanned["state"][k])
                for k in baseline["state"])
        info=(baseline["info"]==scanned["info"])
        rng=rng_equal(baseline["rng"],scanned["rng"])
        continuation=np.array_equal(baseline["next16"],scanned["next16"])
        if not all((six,info,rng,continuation)):
            raise AssertionError(
                f"Continuous observation changed final PCC result: {dataset}/{seed}")

        checkpoint=scan.stop_checkpoint
        prefix=(identical_raw(checkpoint["state"],terminal["state"])
                and rng_equal(checkpoint["rng"],terminal["rng"]))
        if not prefix:
            raise AssertionError(
                f"Early-stop native prefix/RNG mismatch: {dataset}/{seed}")

        predicted_stop=scan.shadow_stop
        actual_stop=(stopped["info"]["effective_iterations"]
                     if stopped["info"]["stop_reason"]=="early_stop" else None)
        if predicted_stop!=actual_stop:
            raise AssertionError(
                f"Shadow stopping time differs: actual={actual_stop}, "
                f"shadow={predicted_stop}")
        if not 0<stopped["info"]["effective_iterations"]<=MAX_ITER:
            raise AssertionError("Invalid native stop iteration")
        unlabeled=(labels==-1)
        stop_labels=stopped["state"]["predictions"]
        end_labels=scanned["state"]["predictions"]
        disagreements=int(np.count_nonzero(
            stop_labels[unlabeled]!=end_labels[unlabeled]))
        correct_stop=float(np.mean(
            stop_labels[unlabeled]==truth[unlabeled]))
        correct_end=float(np.mean(
            end_labels[unlabeled]==truth[unlabeled]))
        report["cases"].append(dict(
            seed=seed,n_nodes=len(labels),n_particles=int(np.count_nonzero(
                labels!=-1)),n_classes=len(np.unique(truth)),
            stop_reason=stopped["info"]["stop_reason"],
            stop_iteration=stopped["info"]["effective_iterations"],
            native_stop_info=stopped["info"],
            no_stop_info=scanned["info"],
            final_six_arrays_exact=six,
            final_stop_info_exact=info,
            final_full_rng_exact=rng,
            final_next16_exact=continuation,
            true_native_prefix_exact=prefix,
            first_shadow_stop_exact=True,
            classification_disagreement_after_continuation=disagreements,
            stop_unlabeled_accuracy=correct_stop,
            continued_unlabeled_accuracy=correct_end,
            times_seconds={
                "baseline":baseline["seconds"],
                "native_stop":stopped["seconds"],
                "continuous_stream":scanned["seconds"],
            },
            dynamics=dynamics))
        Path(output).write_text(json.dumps(report,indent=2,
            sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
        print(
            f"{dataset}/{seed}: stop={stopped['info']['stop_reason']}@"
            f"{stopped['info']['effective_iterations']}; "
            f"final+prefix RNG parity PASS; "
            f"6002 callbacks; {dynamics['quiet_but_hidden_windows']}/"
            f"{dynamics['quiet_endpoint_windows']} quiet windows conceal "
            f"intermediate reversals; {disagreements} post-stop disagreements",
            flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=("iris","wine","breast_cancer","digits"),
                   required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    evaluate_dataset(a.dataset,a.output)


if __name__=="__main__":
    main()
