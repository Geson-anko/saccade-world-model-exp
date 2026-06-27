# syntax=docker/dockerfile:1.7
#
# saccade-world-model-exp 開発環境イメージ。
#   - debian:13-slim ベース (最小)。
#   - uv / just / gh を /usr/local 配下に root で固定インストール。Python は uv が
#     pinned バージョン (.python-version = 3.13) を引く。
#   - host UID/GID 一致の `dev` ユーザーで実行 (bind 経由の成果物が host 所有になる)。
#   - ソースコードは image に焼かない。host repo を /workspace に bind mount し、
#     uv sync / pre-commit install は .devcontainer/post-create.sh が冪等に行う。
#   - GPU: CUDA はベースイメージに含めない。GPU デバイスは compose の device 予約で
#     渡し、nvidia-container-toolkit が host ドライバ (nvidia-smi 等) を inject する。
#     CUDA ランタイムは将来導入するフレームワーク (torch 等) の wheel が同梱する想定。

FROM debian:13-slim

ARG USER_UID=1000
ARG USER_GID=1000

ENV HOME=/home/dev
# $HOME/.local/bin は uv tool / Claude Code CLI の既定インストール先。
ENV PATH=/usr/local/bin:/home/dev/.local/bin:$PATH
# cache volume と workspace が別 FS のとき uv の hardlink 警告を避ける。
ENV UV_LINK_MODE=copy

# build-essential は将来 wheel が無い依存を source build する場合の保険。
# locales は VS Code 等が転送する en_US.UTF-8 を生成し bash 起動時の警告を抑止する。
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        unzip \
        tar \
        procps \
        openssh-client \
        bash-completion \
        locales \
        sudo \
 && sed -i 's/^# *\(en_US.UTF-8 UTF-8\)/\1/' /etc/locale.gen \
 && locale-gen \
 && rm -rf /var/lib/apt/lists/*

# uv + just を /usr/local/bin に固定インストール。
RUN set -eux; \
    curl -LsSf https://astral.sh/uv/install.sh \
      | UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh; \
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
      | bash -s -- --to /usr/local/bin

# GitHub CLI (gh) を公式 apt リポジトリから。Claude からの `git push` / `gh pr` 用。
RUN set -eux; \
    install -d -m 0755 /etc/apt/keyrings; \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      -o /etc/apt/keyrings/githubcli-archive-keyring.gpg; \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends gh; \
    rm -rf /var/lib/apt/lists/*

# bash 補完 (`docker exec dev bash` の非 login 対話 shell でも効くように)。
RUN set -eux; \
    uv generate-shell-completion bash > /etc/bash_completion.d/uv; \
    just --completions bash > /etc/bash_completion.d/just; \
    echo '[ -f /usr/share/bash-completion/bash_completion ] && . /usr/share/bash-completion/bash_completion' \
      >> /etc/bash.bashrc

# host UID/GID 一致の dev ユーザー。named volume のマウント先を dev 所有で先行作成
# しておく (Docker は初回マウント時に target dir の所有権を継承するため)。
RUN groupadd -g ${USER_GID} dev \
 && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash dev \
 && mkdir -p \
      /home/dev/.cache/uv \
      /home/dev/.cache/pre-commit \
      /home/dev/.claude \
      /home/dev/.config/gh \
 && chown -R dev:dev /home/dev \
 && echo 'dev ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/dev \
 && chmod 0440 /etc/sudoers.d/dev

USER dev
WORKDIR /workspace

# Claude Code CLI を公式 installer で導入 ($HOME/.local/bin)。失敗しても build は継続。
RUN curl -fsSL https://claude.ai/install.sh | bash || echo "warning: Claude Code CLI install failed; use the VS Code extension instead."

CMD ["sleep", "infinity"]
