#!/usr/bin/env bash
# .devcontainer/initialize.sh
#
# devcontainer の initializeCommand から host 側で実行される冪等スクリプト。
# cwd は workspace root (repo ルート)。compose 方式の devcontainer では
# updateRemoteUserUID が効かないため、host の uid/gid を compose の変数補間用
# .env (.devcontainer/.env) に書き出して build-arg 経由で揃える。
set -euo pipefail

ENV_FILE=".devcontainer/.env"
touch "$ENV_FILE"

# DATA_DIR (host 上のデータセット位置) を検証する。compose が ${DATA_DIR} を
# container の /data に read-only マウントするため、未設定だとマウントできない。
# 環境ごとに異なるので .env に設定する想定 (環境変数で渡してもよい)。
if [[ -z "${DATA_DIR:-}" ]]; then
  DATA_DIR="$(sed -n 's/^DATA_DIR=//p' "$ENV_FILE" | tail -n1)"
fi
if [[ -z "${DATA_DIR:-}" ]]; then
  echo "error: DATA_DIR が未設定です。${ENV_FILE} に 'DATA_DIR=/path/to/datasets' を追加してください (host 上のデータセットの場所)。" >&2
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "error: DATA_DIR='${DATA_DIR}' が host 上に存在しないか、ディレクトリではありません。" >&2
  exit 1
fi
echo "info: DATA_DIR=${DATA_DIR} (container では /data に read-only マウント)"

# 既存の HOST_UID / HOST_GID 行を消して追記し直す (冪等)。
sed -i '/^HOST_UID=/d; /^HOST_GID=/d' "$ENV_FILE"
{
  echo "HOST_UID=$(id -u)"
  echo "HOST_GID=$(id -g)"
} >>"$ENV_FILE"

echo "info: wrote HOST_UID=$(id -u) HOST_GID=$(id -g) to ${ENV_FILE}"
