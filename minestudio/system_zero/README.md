# System Zero

System Zero contains bounded, verifier-driven Minecraft mechanics. It receives an already-known operation and executes it through ordinary player controls. It does not explore, select hidden resources, use hidden maps, or decide strategic objectives.

The first registry exposes:

- `MINE_AND_COLLECT`
- `COLLECT_KNOWN_DROP`
- `CRAFT_RECIPE`
- `EQUIP_ITEM`
- `PLACE_BLOCK`
- `RECLAIM_BLOCK`
- `PLACE_STATION`

Every registry entry publishes parameters, preconditions, execution backend, success and failure predicates, a step timeout, abort conditions, and a structured result. `ExecutionRouter` sends unknown or visually uncertain operations to System One. Missing deterministic resources remain a `PRECONDITION_FAILED` result for System Two to handle.

`LegacyClosedLoopBackend` preserves the currently validated controllers while the library interface becomes stable. It accepts only actor-visible observation state and target information explicitly supplied in the request.

Run the contract tests with:

```text
python -m unittest discover -s tests -p test_system_zero.py -v
```

Run the controlled MineStudio smoke with:

```text
python run/system_zero_smoke_test.py
```

Set `MINESTUDIO_SYSTEM_ZERO_ATTEMPTS=100` for the controlled reliability campaign after a one-attempt live smoke passes.
