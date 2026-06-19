#!/usr/bin/env bash
# Build the MultiPL-E code-execution Apptainer image (aarch64 / GH200).
#
# Run this on a node WITH internet access (login node). The resulting .sif bakes in all
# language toolchains, so it needs no network when executing code on compute nodes.
#
# Usage:
#   bash build_executor.sh [output.sif]
#
# Env overrides:
#   MULTIPL_EXEC_SIF   output path (default: <this dir>/multipl-exec.sif)
#   APPTAINER_TMPDIR   build scratch dir (default: <this dir>/.apptainer-tmp)
#   APPTAINER_CACHEDIR layer cache dir   (default: <this dir>/.apptainer-cache)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/multipl-exec.def"
OUT_SIF="${1:-${MULTIPL_EXEC_SIF:-${SCRIPT_DIR}/multipl-exec.sif}}"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRIPT_DIR}/.apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRIPT_DIR}/.apptainer-tmp}"
export TMPDIR="${APPTAINER_TMPDIR}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo ">> Building ${OUT_SIF}"
echo ">>   from   ${DEF_FILE}"

# %files paths in the .def are relative to the build context, so build from this dir.
cd "${SCRIPT_DIR}"

# --fakeroot lets %post run apt/rustup unprivileged. If your site lacks fakeroot
# (subuid/subgid) support, retry with: apptainer build --userns ... (proot fallback).
apptainer build --fakeroot "${OUT_SIF}" "${DEF_FILE}"

echo ">> Built ${OUT_SIF}"
echo ">> Self-test (toolchains):"
apptainer test "${OUT_SIF}" || true
