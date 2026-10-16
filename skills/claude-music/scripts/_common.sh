# shellcheck shell=bash
# Helpers sourced by the shell wrappers. Not executable on its own.

# First Python 3 interpreter that actually runs. On Windows, `python3` and
# `python` can be the Microsoft Store alias stub: present on PATH, but it only
# prints an install hint and exits 9009 (49 after truncation).
find_python() {
    local cand
    for cand in python3 python py; do
        if command -v "$cand" &>/dev/null &&
           "$cand" -c "import sys; sys.exit(sys.version_info[0] != 3)" &>/dev/null; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

# Escape a value for use inside a JSON string literal. Windows paths carry
# backslashes, which are invalid JSON escapes when echoed raw.
json_escape() {
    local s="${1//\\/\\\\}"
    printf '%s' "${s//\"/\\\"}"
}

# ace_step_dir from config.json, or nothing when the file or Python is missing.
# utf-8-sig because a config written by PowerShell starts with a BOM.
read_ace_step_dir() {
    local config="$1" py
    [ -f "$config" ] || return 0
    py=$(find_python) || return 0
    "$py" -c "import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8-sig'))['ace_step_dir'])" \
        "$config" 2>/dev/null | tr -d '\r' || true
}
