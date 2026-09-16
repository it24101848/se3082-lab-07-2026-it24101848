# Exercise 07: Summary Comparison

## 1. Feature comparison

| Program | Collectives used | Array allocation | Root manual summation loop? | Final result available on |
|---|---|---|---|---|
| `sum_bcast.c` (Ex.1) | `MPI_Bcast` + `MPI_Send`/`MPI_Recv` | Every process allocates the **full** N-element array | Yes — root loops over `size-1` `MPI_Recv` calls | Root only |
| `sum_scatter.c` (Ex.2) | `MPI_Scatter` + `MPI_Send`/`MPI_Recv` | Only root allocates the full array; every process allocates a `chunk_size` buffer | Yes — same manual receive loop | Root only |
| `sum_gather.c` (Ex.3) | `MPI_Scatter` + `MPI_Gather` | Only root allocates the full array; every process allocates a `chunk_size` buffer | Yes — root loops over `all_sums[]` to add them | Root only |
| `sum_reduce.c` (Ex.4) | `MPI_Scatter` + `MPI_Reduce` | Only root allocates the full array; every process allocates a `chunk_size` buffer | No — `MPI_Reduce` does the summation | Root only |
| `sum_allreduce.c` (Ex.5) | `MPI_Scatter` + `MPI_Allreduce` | Only root allocates the full array; every process allocates a `chunk_size` buffer | No | **All** processes |
| `sum_scan.c` (Ex.6) | `MPI_Scatter` + `MPI_Scan` | Only root allocates the full array; every process allocates a `chunk_size` buffer | No | **Different value per rank** (prefix sum); last rank holds the global total |

## 2. Timing runs

> These programs require an MPI toolchain (`mpicc`/`mpirun`) which is **not installed in this environment**, so the numbers below were not measured here. Run the following on a machine with MPI installed and fill in the table:
>
> ```
> make run NP=2
> make run NP=4
> make run NP=8
> ```
>
> Each program prints a `Time = ...` line; record it below.

| Program | np=2 | np=4 | np=8 |
|---|---|---|---|
| Bcast + Send/Recv | | | |
| Scatter + Send/Recv | | | |
| Scatter + Gather | | | |
| Scatter + Reduce | | | |
| Scatter + Allreduce | | | |
| Scatter + Scan | | | |

**Expected trend and why:**

* `sum_bcast.c` should be the **slowest**, because `MPI_Bcast` moves the *entire* 1,000,000-element array to every process (O(N·P) total data movement) instead of just the O(N) total that a proper partition needs, and the trailing collection is a serial O(P) chain of `MPI_Send`/`MPI_Recv` calls on root.
* Switching to `MPI_Scatter` (Ex.2) cuts communication volume roughly by a factor of P, since each process only receives its own `N/size` chunk instead of the full array — this alone should give the biggest speed jump.
* Replacing the manual `Send`/`Recv` loop with `MPI_Gather` (Ex.3) does not change the amount of data moved (still O(P) `long long` values) but lets the MPI implementation choose a more efficient communication pattern than a naive serial loop, so it is usually about the same speed or slightly faster.
* `MPI_Reduce` (Ex.4) and `MPI_Allreduce` (Ex.5) use a tree/butterfly algorithm that combines partial sums in O(log P) communication steps instead of O(P), so they should be faster than Gather as P grows, with Allreduce doing slightly more work than Reduce (every process ends up with the answer) but often implemented just as efficiently via recursive doubling.
* `MPI_Scan` (Ex.6) does asymptotically the same amount of communication as Allreduce (it's a prefix variant of the same recursive-doubling family) so its timing should be close to Reduce/Allreduce.
* As process count increases, the gap between the Bcast version and the Reduce/Allreduce/Scan versions should widen, since Bcast's cost grows with both N and P while the others' communication volume no longer depends on N after the initial scatter.

## 3. Thinking question: `MPI_Scan` vs `MPI_Allreduce`

Use `MPI_Scan` instead of `MPI_Allreduce` whenever each process needs to know its **position relative to the others**, not just the grand total.

**Concrete example — parallel file write with global offsets:**
Suppose P processes each produce a variable-length chunk of records that must be written into one shared output file, with each process's records placed contiguously and in rank order. Every process computes `local_count` (number of records it produced). Using `MPI_Scan` with `MPI_SUM` on `local_count` gives each rank its `prefix_sum`; subtracting `local_count` gives `sum_before_me`, which is exactly the **global byte/record offset** at which that process should start writing (e.g., via `MPI_File_write_at` or `fseek`). No process needs the grand total to compute its own offset — only the count from ranks before it — so `MPI_Scan` gives exactly the right (and cheaper-to-reason-about) information in a single call. `MPI_Allreduce` would give every process the *same* total record count, which tells nobody where to start writing; you'd have to fall back to a separate, more complex scheme to compute individual offsets.
