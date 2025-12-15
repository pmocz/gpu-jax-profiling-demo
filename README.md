# GPU JAX Profiling Demo

Philip Mocz (2025)

Profiling JAX on Rusty.

👉 See [slides](https://pmocz.github.io/gpu-jax-profiling-demo/)

For more info, see:

* https://jax-ml.github.io/scaling-book/profiling/
* https://docs.jax.dev/en/latest/profiling.html#
* https://www.youtube.com/watch?v=pPTayTD2rOE


# XProf/TensorBoard

Submit the job to run on Rusty

```bash
sbatch sbatch_rusty_tensorboard.sh
```

Then, download the `logs/` folder to your machine,
and view it with tensorboard

```bash
tensorboard --logdir=logs
```

For reference, the code will output something like:

```console
Time: 0.0046s
JIT Time: 0.0087s
```

Let's look at this and see what is happening ...

## Nvidia Nsight Systems

Now submit the job

```bash
sbatch sbatch_rusty_nsys.sh
```

Then, download the `jax_trace.nsys-rep` file to your machine,
and view it with Nvidia Nsight Systems

## Update Presentation

```bash
python -m pip install git+https://gitlab.com/da_doomer/markdown-slides.git
```

```bash
mdslides ./presentation.md --include media
```

```bash
open ./presentation/index.html
```
