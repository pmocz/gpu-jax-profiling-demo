import jax
import time

"""
GPU JAX Profiling Demo on Rusty
With TensorBoard
Philip Mocz (2025)
"""


def main():
    with jax.profiler.trace("logs"):
        # generate large random matrix
        key = jax.random.key(0)
        x = jax.random.normal(key, (8192, 8192))

        # warm-up
        y = x @ x + 1

        # time
        start = time.time()
        y = x @ x + 1
        y.block_until_ready()
        elapsed = time.time() - start
        print(f"Time: {elapsed:.4f}s")

        # JIT version
        @jax.jit
        def my_func(a, b):
            return a @ b + 1

        # warm-up
        y = my_func(x, x)

        # time
        start = time.time()
        y = my_func(x, x)
        y.block_until_ready()
        elapsed = time.time() - start
        print(f"JIT Time: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
