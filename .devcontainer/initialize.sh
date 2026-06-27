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

# 既存の HOST_UID / HOST_GID 行を消して追記し直す (冪等)。
sed -i '/^HOST_UID=/d; /^HOST_GID=/d' "$ENV_FILE"
{
  echo "HOST_UID=$(id -u)"
  echo "HOST_GID=$(id -g)"
} >>"$ENV_FILE"

echo "info: wrote HOST_UID=$(id -u) HOST_GID=$(id -g) to ${ENV_FILE}"
