# =========================
# path: scripts/setup_and_run.sh
# =========================
#!/usr/bin/env bash
set -euo pipefail
APP_ENTRY="tools/describe_objects.py"
REQ_FILE="requirements.txt"
VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN=""
OUT_DIR_DEFAULT="out"
DIM="$(printf '\033[2m')"; RED="$(printf '\033[31m')"; GRN="$(printf '\033[32m')"; NC="$(printf '\033[0m')"

usage() {
  cat <<'HLP'
Uso:
  scripts/setup_and_run.sh setup [--skip-tests]
  scripts/setup_and_run.sh cache-models [--lite]
  scripts/setup_and_run.sh run-img <imagen> [--out OUT] [--lite] [--cpu] [--score S] [--max-side N]
  scripts/setup_and_run.sh run-dir <carpeta> [--out OUT] [--lite] [--cpu] [--score S] [--max-side N]
  scripts/setup_and_run.sh run-vid <video> [--out OUT] [--lite] [--cpu] [--fps F] [--score S] [--max-side N]
  scripts/setup_and_run.sh run-cam [--out OUT] [--lite] [--cpu] [--fps F] [--max-side N]
  scripts/setup_and_run.sh docker-build [--tag objdesc:cpu]
  scripts/setup_and_run.sh docker-run <run-img|run-dir|run-vid> <ruta> [--out OUT] [--lite] [--cpu] [--fps F] [--score S] [--max-side N] [--tag objdesc:cpu]
HLP
}
log(){ echo -e "${DIM}[$(date +%H:%M:%S)]${NC} $*"; }
ok(){ echo -e "${GRN}✔${NC} $*"; }
err(){ echo -e "${RED}✖${NC} $*" >&2; }
ensure_file(){ [[ -f "$1" ]] || { err "Falta archivo: $1"; exit 1; }; }
detect_python(){
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then :; elif command -v python >/dev/null 2>&1; then PYTHON_BIN="python"; else err "Python no encontrado."; exit 1; fi
  ok "Python: $($PYTHON_BIN --version)"
}
activate_venv(){
  [[ -d "$VENV_DIR" ]] || { log "Creando venv $VENV_DIR"; "$PYTHON_BIN" -m venv "$VENV_DIR"; }
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PIP_BIN="pip"
  ok "Venv activado: $(python --version)"
}
install_requirements(){
  ensure_file "$REQ_FILE"
  "$PIP_BIN" install --upgrade pip
  "$PIP_BIN" install -r "$REQ_FILE"
  ok "Dependencias listas"
}
run_tests_if_any(){
  if [[ -d tests ]]; then
    if python -c "import pytest" >/dev/null 2>&1; then
      log "Ejecutando tests"; pytest -q; ok "Tests OK"
    else log "pytest no instalado, omitiendo tests"; fi
  else log "No hay tests/"; fi
}
cache_models(){
  local lite="${1:-0}"
  python - <<PY
from transformers import pipeline
det = pipeline('object-detection', model=('hustvl/yolos-tiny' if $lite else 'facebook/detr-resnet-50'), device=-1)
cap = pipeline('image-to-text', model=('Salesforce/blip-image-captioning-base' if $lite else 'Salesforce/blip-image-captioning-large'), device=-1)
print('cached', det.model.name_or_path, cap.model.name_or_path)
PY
  ok "Modelos cacheados"
}
ensure_app(){ ensure_file "$APP_ENTRY"; }

cmd_setup(){
  local skip_tests=0
  while [[ $# -gt 0 ]]; do case "$1" in --skip-tests) skip_tests=1; shift ;; *) err "Flag desconocida: $1"; usage; exit 1;; esac; done
  detect_python; activate_venv; install_requirements; [[ $skip_tests -eq 0 ]] && run_tests_if_any; ok "Setup completo"
}
cmd_cache_models(){ local lite=0; while [[ $# -gt 0 ]]; do case "$1" in --lite) lite=1; shift;; *) err "Flag desconocida: $1"; usage; exit 1;; esac; done; detect_python; activate_venv; install_requirements; cache_models "$lite"; }
cmd_run_img(){
  ensure_app; local image="${1:-}"; shift || true; [[ -z "$image" ]] && { err "Falta <imagen>"; usage; exit 1; }
  detect_python; activate_venv
  local out="$OUT_DIR_DEFAULT" lite="" cpu="" score="0.5" maxside=""
  while [[ $# -gt 0 ]]; do
    case "$1" in --out) out="$2"; shift 2;; --lite) lite="--lite"; shift;; --cpu) cpu="--cpu"; shift;; --score) score="$2"; shift 2;; --max-side) maxside="--max-side $2"; shift 2;; *) err "Flag desconocida: $1"; usage; exit 1;; esac
  done
  mkdir -p "$out"
  python "$APP_ENTRY" --input "$image" --out "$out" $lite $cpu --score "$score" $maxside
  ok "Listo → $out/"
}
cmd_run_dir(){
  ensure_app; local dir="${1:-}"; shift || true; [[ -z "$dir" ]] && { err "Falta <carpeta>"; usage; exit 1; }
  detect_python; activate_venv
  local out="$OUT_DIR_DEFAULT" lite="" cpu="" score="0.5" maxside=""
  while [[ $# -gt 0 ]]; do
    case "$1" in --out) out="$2"; shift 2;; --lite) lite="--lite"; shift;; --cpu) cpu="--cpu"; shift;; --score) score="$2"; shift 2;; --max-side) maxside="--max-side $2"; shift 2;; *) err "Flag desconocida: $1"; usage; exit 1;; esac
  done
  mkdir -p "$out"
  python "$APP_ENTRY" --input "$dir" --out "$out" $lite $cpu --score "$score" $maxside
  ok "Listo → $out/"
}
cmd_run_vid(){
  ensure_app; local vid="${1:-}"; shift || true; [[ -z "$vid" ]] && { err "Falta <video>"; usage; exit 1; }
  detect_python; activate_venv
  local out="$OUT_DIR_DEFAULT" lite="" cpu="" score="0.5" fps="1.0" maxside=""
  while [[ $# -gt 0 ]]; do
    case "$1" in --out) out="$2"; shift 2;; --lite) lite="--lite"; shift;; --cpu) cpu="--cpu"; shift;; --score) score="$2"; shift 2;;
                     --fps) fps="$2"; shift 2;; --max-side) maxside="--max-side $2"; shift 2;; *) err "Flag desconocida: $1"; usage; exit 1;; esac
  done
  mkdir -p "$out"
  python "$APP_ENTRY" --video "$vid" --out "$out" $lite $cpu --score "$score" --fps "$fps" $maxside
  ok "Listo → $out/"
}
cmd_run_cam(){
  ensure_app; detect_python; activate_venv
  local out="$OUT_DIR_DEFAULT" lite="" cpu="" fps="1.0" maxside=""
  while [[ $# -gt 0 ]]; do
    case "$1" in --out) out="$2"; shift 2;; --lite) lite="--lite"; shift;; --cpu) cpu="--cpu"; shift;;
                     --fps) fps="$2"; shift 2;; --max-side) maxside="--max-side $2"; shift 2;; *) err "Flag desconocida: $1"; usage; exit 1;; esac
  done
  mkdir -p "$out"
  python "$APP_ENTRY" --webcam --out "$out" $lite $cpu --fps "$fps" $maxside
}
cmd_docker_build(){ local tag="objdesc:cpu"; while [[ $# -gt 0 ]]; do case "$1" in --tag) tag="$2"; shift 2;; *) err "Flag desconocida: $1"; usage; exit 1;; esac; done; ensure_file "Dockerfile"; docker build -t "$tag" .; ok "Docker build OK"; }
cmd_docker_run(){
  local sub="${1:-}"; shift || true
  local path="${1:-}"; if [[ "$sub" != "run-cam" ]]; then [[ -z "$path" ]] && { err "Falta <ruta>"; usage; exit 1; }; fi
  local tag="objdesc:cpu" out="$OUT_DIR_DEFAULT" lite="" cpu="--cpu" score="0.5" fps="1.0" maxside=""
  while [[ $# -gt 0 ]]; do
    case "$1" in --tag) tag="$2"; shift 2;; --out) out="$2"; shift 2;; --lite) lite="--lite"; shift;;
                   --cpu) cpu="--cpu"; shift;; --score) score="$2"; shift 2;; --fps) fps="$2"; shift 2;; --max-side) maxside="--max-side $2"; shift 2;;
                   *) shift;; esac
  done
  mkdir -p "$out"
  case "$sub" in
    run-img) docker run --rm -v "$PWD:/app" "$tag" --input "/app/$path" --out "/app/$out" $lite $cpu --score "$score" $maxside ;;
    run-dir) docker run --rm -v "$PWD:/app" "$tag" --input "/app/$path" --out "/app/$out" $lite $cpu --score "$score" $maxside ;;
    run-vid) docker run --rm -v "$PWD:/app" "$tag" --video "/app/$path" --out "/app/$out" $lite $cpu --score "$score" --fps "$fps" $maxside ;;
    *) err "Subcomando docker-run inválido: $sub"; usage; exit 1 ;;
  esac
  ok "Docker run OK → $out/"
}
main(){
  [[ $# -lt 1 ]] && { usage; exit 1; }
  case "$1" in
    help|-h|--help) usage ;;
    setup) shift; cmd_setup "$@" ;;
    cache-models) shift; cmd_cache_models "$@" ;;
    run-img) shift; cmd_run_img "${1:-}" "${@:2}" ;;
    run-dir) shift; cmd_run_dir "${1:-}" "${@:2}" ;;
    run-vid) shift; cmd_run_vid "${1:-}" "${@:2}" ;;
    run-cam) shift; cmd_run_cam "$@" ;;
    docker-build) shift; cmd_docker_build "$@" ;;
    docker-run) shift; cmd_docker_run "$@" ;;
    *) err "Comando desconocido: $1"; usage; exit 1 ;;
  esac
}
main "$@"