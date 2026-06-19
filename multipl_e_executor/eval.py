import logging
import uuid
from pathlib import Path

import eval_cpp

# import eval_julia
import eval_java

# import eval_lua
# import eval_racket
import eval_javascript
import eval_php

# import eval_lua
import eval_python

# import eval_adb
import eval_rust
import eval_sh

# Added for the extended language set (Go / Ruby / TypeScript).
import eval_go
import eval_ruby
import eval_ts

# import eval_dlang
# import eval_julia
# import eval_r
# import eval_fs
# import eval_ocaml
# import eval_matlab
# import eval_hs
# import eval_elixir
# import eval_clj
# import eval_v
# import eval_lean
# import eval_dart


EVALUATORS = {
    # "ada": (eval_adb.eval_script, ".adb"),
    # "rb": (eval_ruby.eval_script, ".rb"),
    # "go_test.go": (eval_go.eval_script, ".go"),
    # "lua": (eval_lua.eval_script, ".lua"),
    "python": (eval_python.eval_script, ".py"),
    "py": (eval_python.eval_script, ".py"),
    "notypes.py": (eval_python.eval_script, ".py"),
    # "julia": (eval_julia.eval_script, ".jl"),
    "java": (eval_java.eval_script, ".java"),
    "rust": (eval_rust.eval_script, ".rs"),
    "rs": (eval_rust.eval_script, ".rs"),
    "sh": (eval_sh.eval_script, ".sh"),
    # "swift": (eval_swift.eval_script, ".swift"),
    # "lua": (eval_lua.eval_script, ".lua"),
    # "racket": (eval_racket.eval_script, ".rkt"),
    # "rkt": (eval_racket.eval_script, ".rkt"),
    "javascript": (eval_javascript.eval_script, ".js"),
    "js": (eval_javascript.eval_script, ".js"),
    "cpp": (eval_cpp.eval_script, ".cpp"),
    "php": (eval_php.eval_script, ".php"),
    "rb": (eval_ruby.eval_script, ".rb"),
    "ts": (eval_ts.eval_script, ".ts"),
    # Go's dataset `language` field is literally "go_test.go"; the file MUST end in
    # `_test.go` or `go test` runs zero tests and exits 0 (silent false pass).
    "go_test.go": (eval_go.eval_script, "_test.go"),
    # "cs": (eval_cs.eval_script, ".cs"),
    # "ts": (eval_ts.eval_script, ".ts"),
    # "humaneval_to_dlang.py": (eval_dlang.eval_script, ".d"),
    # "d": (eval_dlang.eval_script, ".d"),
    # "r": (eval_r.eval_script, ".r"),
    # "humaneval_to_r.py": (eval_r.eval_script, ".r"),
    # "jl": (eval_julia.eval_script, ".jl"),
    # "fs": (eval_fs.eval_script, ".fsx"),
    # "ml": (eval_ocaml.eval_script, ".ml"),
    # "m": (eval_matlab.eval_script, ".m"),
    # "hs": (eval_hs.eval_script, ".hs"),
    # "elixir": (eval_elixir.eval_script, ".exs"),
    # "clj": (eval_clj.eval_script, ".clj"),
    # "coq": (eval_v.eval_script, ".v"),
    # "lean": (eval_lean.eval_script, ".lean"),
    # "dart": (eval_dart.eval_script, ".dart"),
}

logger = logging.getLogger(__name__)


def check_correctness(language: str, completion: str, test: str):
    if language in EVALUATORS:
        (eval_script, file_ext) = EVALUATORS[language]
    else:
        eval_module = __import__(f"eval_{language}" if language != "go_test.go" else "eval_go")
        eval_script = eval_module.eval_script

    program = completion + "\n" + test  # note completion should include original prompt

    # temp file for execution; Lambda requires using /tmp dir for ephemeral storage
    unique_id = str(uuid.uuid4())
    tmp_dir = Path("/tmp")
    temp_path = tmp_dir / f"eval_{unique_id}{file_ext}"

    try:
        temp_path.write_text(program)
        result = eval_script(temp_path)

        if result["stderr"]:
            logger.error("Execution stderr: %s", result["stderr"])

        if result["stdout"]:
            logger.info("Execution stdout: %s", result["stdout"])

        return {
            "tested_completion": program,
            "exit_code": result["exit_code"],
            "status": result["status"],
        }
    finally:
        # Clean up the temporary file
        if temp_path.exists():
            temp_path.unlink()
