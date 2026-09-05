# Running RTUS on Kubernetes

The image is self-contained: package, configs and traiNNer are baked in. A Job
therefore needs to supply **data and a config name**, and nothing else.

That is the point of this layout. Previously a run meant authoring a Job with
inline shell, and getting the config onto the node by hand — which is how one
config in this project's history became unrecoverable and had to be
reconstructed from logs.

## Starting a run

Copy `overlays/example`, change three things, apply:

```
cp -r k8s/overlays/example k8s/overlays/my-run
# 1. nameSuffix       -- what this run is called
# 2. the -opt arg     -- which config, by name, from /app/configs
# 3. the data volume  -- your PVC or hostPath
```

The whole overlay is ~30 lines. Everything else — GPU limit, memory ceiling,
shared-memory sizing, restart semantics — is inherited from `base/`.

## Choices in the base worth knowing

**`backoffLimit: 0`.** Training is not idempotent: a retry resumes from
whatever checkpoint the previous attempt wrote. That is usually desirable, but
an automatic retry after an OOM looks like one mysteriously slow run rather
than two failed ones. Fail visibly instead.

**`/dev/shm` sized at 4Gi.** torch dataloader workers communicate through
shared memory and the container default of 64 MB will deadlock a multi-worker
run — with no useful error.

**Memory limit 24Gi.** Dataloader workers cost roughly 1.2 GB RSS each, so
`num_worker_per_gpu` in the config and this limit have to be read together.
Raising one without the other is how a node gets OOM-killed.

**`ttlSecondsAfterFinished: 7 days.** Long enough to read logs after a
weekend, short enough that finished Jobs do not accumulate.

## Listing configs in an image

```
kubectl run rtus-configs --rm -it --restart=Never \
  --image=<your-registry>/rtus-train:0.1.0 -- ls /app/configs
```

## Checking an environment before blaming the model

`rtus-info` reports whether the traiNNer registry resolved, where
`RTUS_DATA_ROOT` points, and which assets are present. It distinguishes "the
model is wrong" from "the environment is incomplete", which otherwise look
alike from a failed Job.

## GitOps

Two ways to consume this, and the choice is yours:

1. **Point a Flux `GitRepository` at this repo** and reconcile
   `k8s/overlays/<run>`. Job specs then live beside the code that runs them,
   and a run is one commit here.
2. **Keep overlays in your cluster repo** and reference the published image.
   Fewer moving parts if your cluster repo is already the single source of
   truth for what runs.

Option 1 is why this directory exists, but it only pays off if your Flux
setup can add a source. Option 2 works with no cluster-side change at all.
