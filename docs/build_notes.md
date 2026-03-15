# Build Notes

## Nature paper

```bash
cd Nat_Paper
./compile_nature.sh
```

The compile script checks that required figure files exist in `Nat_Paper/figures/` before invoking LaTeX.

## Appendix

```bash
cd GitHub/appendix
./compile_appendix.sh
```

The appendix compiles using relative paths to `GitHub/appendix/figures/`.

## Smoke test

```bash
python GitHub/scripts/smoke_test.py --root .
```

This validates core imports, path integrity, SQLite setup, and a tiny n-gram run.
