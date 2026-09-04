#!/usr/bin/env bash

# Download, verify, and extract the complete TAU Urban Audio-Visual Scenes
# 2021 development corpus from the immutable Zenodo record used by PATH-WM.

set -Eeuo pipefail

readonly RECORD_ID="4477542"
readonly BASE_URL="https://zenodo.org/api/records/${RECORD_ID}/files"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly DATASET_ROOT="${REPO_ROOT}/data/tau_urban_av_2021"
readonly RAW_ROOT="${DATASET_ROOT}/raw"
readonly ARCHIVE_ROOT="${PATH_WM_TAU_ARCHIVE_ROOT:-${DATASET_ROOT}/.archives}"
readonly COMPLETE_ROOT="${ARCHIVE_ROOT}/completed"
readonly KEEP_ARCHIVES="${PATH_WM_TAU_KEEP_ARCHIVES:-0}"
readonly MIN_FREE_KIB="${PATH_WM_TAU_MIN_FREE_KIB:-20971520}"

# filename|size in bytes|official MD5 from DOI 10.5281/zenodo.4477542
readonly -a ARCHIVES=(
  "TAU-urban-audio-visual-scenes-2021-development.audio.1.zip|4349202707|186f6273f8f69ed9dbdc18ad65ac234f"
  "TAU-urban-audio-visual-scenes-2021-development.audio.2.zip|4490960726|7fd6bb63127f5785874a55aba4e77aa5"
  "TAU-urban-audio-visual-scenes-2021-development.audio.3.zip|4223486056|61396bede29d7c8c89729a01a6f6b2e2"
  "TAU-urban-audio-visual-scenes-2021-development.audio.4.zip|4172854493|6ddac89717fcf9c92c451868eed77fe1"
  "TAU-urban-audio-visual-scenes-2021-development.audio.5.zip|4185568337|af4820756cdf1a7d4bd6037dc034d384"
  "TAU-urban-audio-visual-scenes-2021-development.audio.6.zip|4215402064|ebd11ec24411f2a17a64723bd4aa7fff"
  "TAU-urban-audio-visual-scenes-2021-development.audio.7.zip|4407222915|2be39a76aeed704d5929d020a2909efd"
  "TAU-urban-audio-visual-scenes-2021-development.audio.8.zip|345864456|972d8afe0874720fc2f28086e7cb22a9"
  "TAU-urban-audio-visual-scenes-2021-development.video.1.zip|4996891312|f89b88f6ff44b109e842bff063612bf1"
  "TAU-urban-audio-visual-scenes-2021-development.video.2.zip|4997056221|433053f76ae028f6c4a86094b91af69c"
  "TAU-urban-audio-visual-scenes-2021-development.video.3.zip|4995965420|974c8e4f3741e065e5c2775f4b9e3ffc"
  "TAU-urban-audio-visual-scenes-2021-development.video.4.zip|4991033216|b558fe62fb8fd9b1864d256ead8a24d1"
  "TAU-urban-audio-visual-scenes-2021-development.video.5.zip|4997081324|fb3036c31e66be1caddb48834e0ea304"
  "TAU-urban-audio-visual-scenes-2021-development.video.6.zip|4991170406|140f331750406eaa16756b5cfdf0a336"
  "TAU-urban-audio-visual-scenes-2021-development.video.7.zip|4996594150|bac47d3da9bffb89318e662c6af68539"
  "TAU-urban-audio-visual-scenes-2021-development.video.8.zip|4992534816|95c231ee549c6e74ab8b57a27047ff71"
  "TAU-urban-audio-visual-scenes-2021-development.video.9.zip|4998329385|294daf6f7de15adcae3a97a8d176eb3a"
  "TAU-urban-audio-visual-scenes-2021-development.video.10.zip|4993017199|c6778f4ddbab163394f7cd26011f1452"
  "TAU-urban-audio-visual-scenes-2021-development.video.11.zip|4993425875|856ecd8fb1df96adcd2cc6928441bbe6"
  "TAU-urban-audio-visual-scenes-2021-development.video.12.zip|4994719640|57b898b19d991fde6df595fda85eb28d"
  "TAU-urban-audio-visual-scenes-2021-development.video.13.zip|4996976020|920562ed29a63abeabfb11b6c0a17a2c"
  "TAU-urban-audio-visual-scenes-2021-development.video.14.zip|4999888160|1febab0738a622dea95926e70d2fc7c4"
  "TAU-urban-audio-visual-scenes-2021-development.video.15.zip|4995886583|d6d57cc85f65b1a589a1d628ca936fc9"
  "TAU-urban-audio-visual-scenes-2021-development.video.16.zip|2286205594|4e1ab47f0d180b818491ef3c3f45ebc6"
)

timestamp() {
  date --iso-8601=seconds
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: scripts/download_tau_urban_av_2021.sh [--list]

With no arguments, downloads all 24 media archives sequentially, resumes partial
downloads, verifies the official byte size and MD5, and extracts into
data/tau_urban_av_2021/raw/{audio,video}. Verified ZIPs are deleted after
successful extraction unless PATH_WM_TAU_KEEP_ARCHIVES=1.

Environment overrides:
  PATH_WM_TAU_ARCHIVE_ROOT  Staging directory for partial ZIPs and markers
  PATH_WM_TAU_KEEP_ARCHIVES Keep verified ZIPs when set to 1
  PATH_WM_TAU_MIN_FREE_KIB  Abort below this free-space floor (default: 20 GiB)
EOF
}

list_archives() {
  local entry filename size checksum
  printf 'filename\tbytes\tmd5\n'
  for entry in "${ARCHIVES[@]}"; do
    IFS='|' read -r filename size checksum <<<"${entry}"
    printf '%s\t%s\t%s\n' "${filename}" "${size}" "${checksum}"
  done
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

check_free_space() {
  local available_kib
  available_kib="$(df -Pk "${DATASET_ROOT}" | awk 'NR == 2 {print $4}')"
  [[ "${available_kib}" =~ ^[0-9]+$ ]] || die "could not determine free disk space"
  (( available_kib >= MIN_FREE_KIB )) || die \
    "only ${available_kib} KiB free; safety floor is ${MIN_FREE_KIB} KiB"
}

download_archive() {
  local filename="$1"
  local expected_size="$2"
  local expected_md5="$3"
  local archive_path="${ARCHIVE_ROOT}/${filename}"
  local partial_path="${archive_path}.part"
  local complete_path="${COMPLETE_ROOT}/${filename}.md5"
  local actual_size actual_md5 verification_path

  if [[ -f "${complete_path}" ]] && [[ "$(<"${complete_path}")" == "${expected_md5}" ]]; then
    log "SKIP ${filename}: verified extraction marker exists"
    return
  fi

  check_free_space
  if [[ -f "${archive_path}" ]]; then
    log "REUSE ${filename}: verified ZIP survived an interrupted extraction"
    verification_path="${archive_path}"
  else
    log "DOWNLOAD ${filename} (${expected_size} bytes)"
    wget \
      --continue \
      --retry-connrefused \
      --waitretry=5 \
      --timeout=60 \
      --tries=0 \
      --progress=dot:giga \
      --output-document="${partial_path}" \
      "${BASE_URL}/${filename}/content"
    verification_path="${partial_path}"
  fi

  actual_size="$(stat --format='%s' "${verification_path}")"
  [[ "${actual_size}" == "${expected_size}" ]] || die \
    "size mismatch for ${filename}: expected ${expected_size}, got ${actual_size}"

  actual_md5="$(md5sum "${verification_path}" | awk '{print $1}')"
  [[ "${actual_md5}" == "${expected_md5}" ]] || die \
    "MD5 mismatch for ${filename}: expected ${expected_md5}, got ${actual_md5}"
  if [[ "${verification_path}" == "${partial_path}" ]]; then
    mv -- "${partial_path}" "${archive_path}"
  fi
  log "VERIFIED ${filename}: ${actual_md5}"

  log "EXTRACT ${filename} -> ${RAW_ROOT}"
  unzip -q -o "${archive_path}" -d "${RAW_ROOT}"
  printf '%s\n' "${expected_md5}" >"${complete_path}"
  log "EXTRACTED ${filename}"

  if [[ "${KEEP_ARCHIVES}" == "0" ]]; then
    rm -- "${archive_path}"
    log "REMOVED verified staging ZIP ${filename}"
  fi
}

main() {
  case "${1:-}" in
    "") ;;
    --list)
      list_archives
      return
      ;;
    --help|-h)
      usage
      return
      ;;
    *)
      usage >&2
      return 2
      ;;
  esac

  require_command awk
  require_command df
  require_command md5sum
  require_command stat
  require_command unzip
  require_command wget
  [[ "${MIN_FREE_KIB}" =~ ^[0-9]+$ ]] || die "PATH_WM_TAU_MIN_FREE_KIB must be an integer"
  [[ "${KEEP_ARCHIVES}" == "0" || "${KEEP_ARCHIVES}" == "1" ]] || die \
    "PATH_WM_TAU_KEEP_ARCHIVES must be 0 or 1"

  mkdir -p "${RAW_ROOT}" "${ARCHIVE_ROOT}" "${COMPLETE_ROOT}"
  log "START TAU full-corpus acquisition (${#ARCHIVES[@]} archives)"
  log "Raw media: ${RAW_ROOT}"
  log "Staging: ${ARCHIVE_ROOT}; keep archives: ${KEEP_ARCHIVES}"

  local entry filename size checksum
  for entry in "${ARCHIVES[@]}"; do
    IFS='|' read -r filename size checksum <<<"${entry}"
    download_archive "${filename}" "${size}" "${checksum}"
  done

  touch "${DATASET_ROOT}/.full_media_download_complete"
  log "COMPLETE: all 24 media archives verified and extracted"
}

trap 'die "failed at line ${LINENO}"' ERR
main "$@"
