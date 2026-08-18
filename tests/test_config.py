"""
test_config.py

The pipeline's configuration is a contract between three files that nothing
enforces at runtime: dvc.yaml names flags, params.yaml supplies values, and the
argparsers decide what is actually accepted. A mismatch does not raise -- it
produces a run that silently did something other than what was configured.

Every test here corresponds to a defect that reached a real run:

- the capacity grid varied `freeze_encoder` per arm while its command never
  passed `--freeze-encoder`, so one arm was a duplicate of another
- `--decoder-channels` received the text "None" from a YAML null and crashed
- benchmark and capacity outputs nested inside the train stage's output
- train.py's argparse defaults drifted from params.yaml, lr by 11x

None of these need torch, a GPU, or the dataset. They run in milliseconds.
"""

import ast
import re

import pytest
from conftest import REPO_ROOT, declared_flags


def stage_bodies(pipeline):
    """Yield (stage_name, body) for every stage, unwrapping `foreach`/`do`."""
    for name, stage in pipeline["stages"].items():
        yield name, stage.get("do", stage)


def module_file_for(cmd):
    """Resolve the script a stage command runs.

    Args:
        cmd: The stage's command string.

    Returns:
        Path to the module's source file, or None if the command is not a
        `python -m` invocation.
    """
    match = re.search(r"python\s+-m\s+([\w.]+)", cmd)
    if not match:
        return None
    return REPO_ROOT / (match.group(1).replace(".", "/") + ".py")


def resolve(text, params):
    """Substitute ${section.key} references from params.yaml into a string."""
    def sub(match):
        section, key = match.group(1), match.group(2)
        return str(params.get(section, {}).get(key, match.group(0)))

    return re.sub(r"\$\{(\w+)\.(\w+)\}", sub, text)


# ---------------------------------------------------------------------------
# dvc.yaml <-> argparser
# ---------------------------------------------------------------------------


def test_every_stage_flag_is_declared_by_its_script(dvc_pipeline):
    """Every --flag a stage passes must exist in that script's argparser.

    Catches a stage command drifting ahead of the script it drives -- a rename
    or a removed flag that argparse would only reject at runtime, hours into a
    grid.
    """
    problems = []
    for name, body in stage_bodies(dvc_pipeline):
        cmd = body["cmd"]
        module_file = module_file_for(cmd)
        if module_file is None:
            continue
        assert module_file.exists(), f"{name}: {module_file} does not exist"

        allowed = declared_flags(module_file)
        used = set(re.findall(r"(?<!\S)(--[a-z0-9][a-z0-9-]*)", cmd))
        for flag in sorted(used - allowed):
            problems.append(f"{name}: passes {flag}, not declared in {module_file.name}")

    assert not problems, "\n".join(problems)


def test_foreach_arms_define_exactly_the_keys_their_command_uses(dvc_pipeline, params, dvc_raw):
    """A grid arm's keys and its command's ${item.*} references must match exactly.

    Both directions matter, and the second is the one that bit us. A key defined
    in params.yaml but never referenced in the command is silently inert: the
    capacity grid set `freeze_encoder: true` on one arm, the command never
    passed it, and that arm trained identically to another. It would have
    produced a plausible-looking null result.
    """
    problems = []
    for name, stage in dvc_pipeline["stages"].items():
        foreach = stage.get("foreach")
        if not foreach:
            continue
        m = re.match(r"\$\{(\w+)\.variants\}", foreach)
        assert m is not None, f"stage {name}: foreach {foreach!r} does not match expected pattern"
        section = m.group(1)
        variants = params[section]["variants"]

        bm = re.search(rf"^  {name}:\n(?:(?:    .*)?\n)*", dvc_raw, re.M)
        assert bm is not None, f"stage {name} not found in dvc.yaml raw text"
        block = bm.group(0)
        used = set(re.findall(r"\$\{item\.(\w+)\}", block))

        for arm_name, arm in variants.items():
            missing = used - set(arm)
            unused = set(arm) - used
            if missing:
                problems.append(f"{name}@{arm_name}: command needs {sorted(missing)}, arm does not define it")
            if unused:
                problems.append(f"{name}@{arm_name}: defines {sorted(unused)}, command never passes it")

    assert not problems, "\n".join(problems)


def test_all_param_references_resolve(dvc_raw, params):
    """Every ${section.key} in dvc.yaml must exist in params.yaml."""
    unresolved = []
    for section, key in set(re.findall(r"\$\{(\w+)\.(\w+)\}", dvc_raw)):
        if section == "item":
            continue
        if section not in params or key not in (params[section] or {}):
            unresolved.append(f"{section}.{key}")
    assert not unresolved, f"unresolved references: {sorted(unresolved)}"


# ---------------------------------------------------------------------------
# Output topology
# ---------------------------------------------------------------------------


def test_no_declared_output_nests_inside_another(dvc_pipeline, params):
    """DVC rejects a tracked output living inside another stage's tracked output.

    It is right to: the outer stage would see itself as modified every time the
    inner one wrote. Checked here because the error surfaces only at
    `dvc repro`, after graph construction.
    """
    outputs = []
    for name, body in stage_bodies(dvc_pipeline):
        for entry in body.get("outs", []):
            path = next(iter(entry)) if isinstance(entry, dict) else entry
            resolved = resolve(path, params).replace("${key}", "ARM").rstrip("/")
            outputs.append((name, resolved))

    for i, (name_a, path_a) in enumerate(outputs):
        for name_b, path_b in outputs[i + 1 :]:
            a, b = path_a + "/", path_b + "/"
            assert not (a.startswith(b) or b.startswith(a)), (
                f"{name_a} output {path_a!r} overlaps {name_b} output {path_b!r}"
            )


def test_mlflow_store_is_never_a_declared_output(dvc_pipeline, params):
    """MLflow's database and artifact store must stay outside DVC's outputs.

    DVC deletes a stage's outputs before running it. Declaring the tracking DB
    would erase every logged run on each `dvc repro` -- silently, since the
    training run afterwards repopulates it with one experiment.
    """
    for name, body in stage_bodies(dvc_pipeline):
        for key in ("outs", "metrics", "plots"):
            for entry in body.get(key, []):
                path = next(iter(entry)) if isinstance(entry, dict) else entry
                resolved = resolve(path, params).lower()
                assert "mlflow" not in resolved and "mlruns" not in resolved, (
                    f"{name} declares MLflow state {resolved!r} as an output"
                )


def test_declared_dependencies_exist(dvc_pipeline):
    """Every non-interpolated dependency path must exist in the repo."""
    missing = []
    for name, body in stage_bodies(dvc_pipeline):
        for dep in body.get("deps", []):
            if dep.startswith("${"):
                continue
            if not (REPO_ROOT / dep).exists():
                missing.append(f"{name}: {dep}")
    assert not missing, "\n".join(missing)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_train_argparser_holds_no_hardcoded_defaults_for_params(params):
    """No training flag may carry a literal default that params.yaml also sets.

    Tests the mechanism, not the values. Asserting the numbers match would pass
    the moment someone re-synced them by hand and start rotting again
    immediately; asserting that the defaults are *read from the file* cannot
    drift at all. The original divergence had lr defaulting to 1e-3 against a
    tuned 9.18e-05, so a manual run looked like a reproduction and was not.
    """
    owned = set()
    for section in ("data", "model", "loss", "train"):
        owned |= {key.replace("_", "-") for key in (params.get(section) or {})}

    tree = ast.parse((REPO_ROOT / "training" / "train.py").read_text())
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if not isinstance(node.args[0].value, str):
            continue
        flag = node.args[0].value.lstrip("-")
        if flag not in owned:
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, (ast.Constant, ast.Tuple, ast.List)):
                offenders.append(f"--{flag} has a literal default; read it from params.yaml instead")

    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize(
    "raw,expected",
    [("None", None), ("null", None), ("", None), ("128", 128), ("256", 256)],
)
def test_optional_int_accepts_yaml_null_placeholders(raw, expected):
    """DVC renders a YAML null into a command string as the text "None".

    argparse with type=int rejects that, which crashed every grid arm leaving
    the value unset -- before a single training step ran.
    """
    parse = _extract_function("optional_int")
    assert parse(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("true", True), ("True", True), ("1", True), ("false", False), ("False", False), ("0", False)],
)
def test_str2bool_accepts_templated_booleans(raw, expected):
    """A store_true flag cannot be templated, so booleans must take a value."""
    parse = _extract_function("str2bool")
    assert parse(raw) is expected


def test_str2bool_rejects_nonsense():
    """A typo in a grid arm should fail loudly, not resolve to False."""
    import argparse

    parse = _extract_function("str2bool")
    with pytest.raises(argparse.ArgumentTypeError):
        parse("maybe")


def _extract_function(name):
    """Load one function from train.py without importing the module.

    train.py imports torch and smp at module scope. These parsers are pure
    string handling, so pulling in a deep-learning stack to test them would make
    a millisecond check depend on the whole environment.

    Args:
        name: Function name to extract.

    Returns:
        The callable.
    """
    import argparse
    from typing import Any, Callable, cast

    tree = ast.parse((REPO_ROOT / "training" / "train.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace: dict[str, Any] = {"argparse": argparse}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<extracted>", "exec"), namespace)
    return cast(Callable[..., Any], namespace[name])
