# Silicon vs silicon nitride for free-electron–light coupling: a simulation study

**Status:** work in progress. The PINEM model, the slab waveguide solver and phase matching are built and tested. There are no results yet on silicon against silicon nitride.

## Research question
For telecom-wavelength waveguides, does Si or Si₃N₄ give stronger electron–light coupling, and does the answer differ between laser-driven (classical) coupling and spontaneous single-photon (quantum) coupling?

## Install
Requires Python 3.11 or newer, because the pinned numpy, scipy, matplotlib and scikit-learn need it.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 -m pytest
```

Activate the environment in every new Terminal window. Otherwise another Python on your computer, such as Anaconda's, may run instead.

Regenerate the figures with `python3 scripts/make_figures.py`.

### Troubleshooting on a Mac
If `import fepc` fails with `No module named 'fepc'` even though the install worked, try these in order.

1. Run `type python`. If it is an alias, it overrides the environment, so use `python3`, as above.
2. Your project folder may be on an iCloud-synced Desktop or Documents folder. iCloud marks the contents of `.venv` as hidden, and recent versions of Python skip hidden `.pth` files, which is how an editable install finds your code. Either keep the project outside iCloud, or link the package into the environment, adjusting `python3.11` to your Python version:

```bash
ln -sfn ../../../../src/fepc .venv/lib/python3.11/site-packages/fepc
```

## Author
Janvi — M.Tech, Solid State & Quantum Materials, IIT Delhi
