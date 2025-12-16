[comment]: # (THEME = moon)
[comment]: # (CODE_THEME = base16/zenburn)

# GPU/JAX Profiling on the Cluster

Philip Mocz (2025)

see: [https://github.com/pmocz/gpu-jax-profiling-demo](https://github.com/pmocz/gpu-jax-profiling-demo)

[comment]: # (!!!)

For this session, we look at
profiling JAX GPU code with:
* XProf/TensorBoard
* NVIDIA Nsight Systems (nsys)
* see also: Perfetto

[comment]: # (!!!)

## Three Layers of JAX performance:

1) Python overhead & dispatch
2) XLA compilation
3) GPU runtime (kernels, memory, transfers)

[comment]: # (!!!)

## Common performance killers:

  - Recompiling (shape/dtype changes)
  - Host↔device transfers
  - Too many small kernels (poor fusion / tiny work)
  - Synchronizations (implicit `.block_until_ready()`, `device_get`, `print`, etc.)

[comment]: # (!!!)

## Warm-Up Pattern

* When profiling, it's usually a good idea to run a function once before actually profiling it, since there is overhead with the initial setup.

[comment]: # (!!!)

## Warm-Up Pattern

* Before profiling, make sure we've done something like `block_until_ready()`

```python
import jax, jax.numpy as jnp

@jax.jit
def step(x):
    return jnp.tanh(x) @ jnp.tanh(x).T

x = jnp.ones((4096, 4096), dtype=jnp.float16)

# Warmup (compile + run)
y = step(x).block_until_ready()

# Steady-state runs
for _ in range(10):
    y = step(x)
y.block_until_ready()
```

[comment]: # (!!!)

## XProf/TensorBoard

* XProf (from OpenXLA) offers a number of tools to analyse and visualize the performance of your model across multiple devices

* TensorBoard is a suite of web applications for inspecting and understanding machine learning experimentation

```console
pip install tensorboard tensorboard-plugin-profile
tensorboard --logdir ./tb-logs --port 6006
```

[comment]: # (!!!)

## Minimal JAX trace capture

```python
import os
import jax
import jax.numpy as jnp

logdir = "./tb-logs"

@jax.jit
def f(x):
    return (x @ x.T).sum()

x = jnp.ones((8192, 8192), dtype=jnp.float16)
f(x).block_until_ready()  # warmup

# Capture a trace window
jax.profiler.start_trace(logdir)
for _ in range(5):
    f(x).block_until_ready()
jax.profiler.stop_trace()
```

Now in TensorBoard: **Profile → Trace Viewer**

[comment]: # (!!!)

## Getting Clean Traces

- Trace only the interesting window (avoid compile warmup)

- Put block_until_ready() at the end of each iteration during tracing

- Keep loops small (1–20 steps)

- Use stable shapes to avoid recompilation noise

[comment]: # (!!!)

## Add named regions (so traces are readable)

```python
import jax
import jax.numpy as jnp

@jax.jit
def train_step(params, batch):
    with jax.named_scope("forward"):
        out = batch @ params
    with jax.named_scope("loss"):
        loss = (out**2).mean()
    with jax.named_scope("backward"):
        grads = jax.grad(lambda p: (batch @ p).mean())(params)
    return loss, grads
```

[comment]: # (!!!)

## Can Profile Device Memory too:

See [https://docs.jax.dev/en/latest/device_memory_profiling.html](https://docs.jax.dev/en/latest/device_memory_profiling.html)

* Understand how program is using GPU/TPU memory
* Debug memory leaks

[comment]: # (!!!)

## Interpreting traces

Look for:

* Compilation spikes (long CPU blocks before GPU work)

* Gaps on GPU timeline (CPU is starving GPU)

* Memcpy / HtoD / DtoH events (host-device transfers)

* Many tiny kernels (overhead & poor utilization)

* Repeated “XLA compile” events (shape churn)

[comment]: # (!!!)

## NVIDIA Nsight Systems (nsys)

Best for: end-to-end timeline across:

  - Python threads
  - CUDA launches
  - kernel execution
  - memory transfers
  - NCCL comms (multi-GPU)
  - CPU-GPU synchronization

[comment]: # (!!!)

## Profile a script with nsys

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --output=nsys_report \
  python your_script.py
```

[comment]: # (!!!)

## Roofline Model

![Roofline Model](media/roofline0.png)

* Memory-bound vs compute-bound
* Fusion moves you right
* Precision moves the roof up
* Helps answer: what optimization is worth it?

[comment]: # (!!!)

## Roofline Model

![Roofline Model](media/roofline.png)

[comment]: # (!!!)

## Tips-and-Tricks

* Do as much heavy-duty work directly on GPU as possible. Minimize large copies between CPU--GPU

* Use single/mixed-precision if possible

* Prefer fused, large kernels

* Profile early - not after months of work

[comment]: # (!!!)

## Tips-and-Tricks

* Asynchronous read/writes:

```python
import orbax.checkpoint as ocp

ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

save_response = ckptr.save(...)

# You can continue with other computations here while saving happens in the background!!!

save_response.wait_until_finished()
```

[comment]: # (!!!)
