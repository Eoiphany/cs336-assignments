# CS336 Spring 2026 Assignment 3: Scaling

For a full description of the assignment, see the assignment handout at
[cs336_assignment3_scaling.pdf](./cs336_assignment3_scaling.pdf).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## For students

Install uv

```sh
uv sync
```

Set `A3_API_KEY` to your 8-digit student ID:

```sh
export A3_API_KEY=06123456
```

The hosted training API is available at:

```text
http://hyperturing.stanford.edu:8000
```

Click here for the [docs](http://hyperturing.stanford.edu:8000/docs) and [dashboard](http://hyperturing.stanford.edu:8000/dashboard).

See [`./examples/client_example.ipynb`](./examples/client_example.ipynb) for an
example of submitting and inspecting training runs.

## For non-students

Install dependencies:

```sh
uv sync --extra server
```

To download tokenized data:

```sh
uv run modal run scripts/1_download_tokenized_data.py
```

To run training directly:

```sh
uv run cs336_scaling/training/run.py
```

To run the API and dispatcher, set:

```sh
DATABASE_URL_PROD="postgresql://..."
DATABASE_URL_DEV="postgresql://..."
INTERNAL_API_KEY="SOMEKEY"
```

Then run:

```sh
DB_ENV=prod uv run fastapi run &
DB_ENV=prod uv run dispatcher &
```

## Offline scaling-law analysis

The local analysis implementation adapts the reference assignment's IsoFLOPs
method to this repository and adds support for the **2026** experiment schema.
It does not call the hosted API or start training.

Reproduce handout §2.1 with the bundled synthetic dataset:

```sh
uv run --extra analysis python cs336_scaling/chinchilla_isoflops.py
```

This writes model-size and token-count plots and a JSON fit under
`runs/isoflops/`, including predictions at `1e23` and `1e24` FLOPs.

For §3, first provide a local JSON array of actual 2026 `/experiments` responses.
Only completed experiments contribute observations. Analyze those observations
and optionally supply a JSON array of full candidate training configurations:

```sh
uv run --extra analysis python cs336_scaling/fit_scaling_laws.py --experiments experiments.json --candidates candidates.json --target-seconds 172800
```

Omit `--candidates` to consider configurations from the observed runs. The output
`runs/scaling/final_submission.local.json` is a local prediction; the script never
submits it. Fits require at least six completed runs with independent variation
in model size and training tokens. An observed fit does not establish that its
48-hour extrapolation is accurate. See [implementation and validation notes](doc/implementation.md)
for assumptions, limitations and verification commands. These same commands work
in Windows CMD, using forward slashes in script paths.
