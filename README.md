# GPU JAX Profiling Demo

Philip Mocz (2025)

Profiling JAX on Rusty.

For more info, see:

* https://jax-ml.github.io/scaling-book/profiling/
* https://docs.jax.dev/en/latest/profiling.html#
* https://www.youtube.com/watch?v=pPTayTD2rOE


# Instuctions

Submit the job to run on Rusty

```bash
sbatch sbatch_rusty.sh
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
