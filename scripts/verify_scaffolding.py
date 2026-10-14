"""Exercise every unopened check/bench worker path, without claiming correctness."""
import contextlib
import io
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_attention_lab.cli import main, parser
from flash_attention_lab.registry import CATALOG, ROOT, load_student

results = []
for e in CATALOG:
    module = load_student(e)
    result = {"exercise": e["id"], "import": "OK"}
    for op in ("check", "bench"):
        parser().parse_args([op, e["id"], "--case", "0"])
        if module.IMPLEMENTED:
            result[op] = "IMPLEMENTED: run explicit acceptance"
            continue
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = main(["_worker", e["id"], "--operation", op, "--case", "0"])
        assert code == 2 and e["id"] in output.getvalue(), (e["id"], op, code, output.getvalue())
        result[op] = {"exit": code, "message": output.getvalue().strip()}
    results.append(result)
print(json.dumps({"student_passes_claimed": 0, "entrypoints": results}, ensure_ascii=False, indent=2))
