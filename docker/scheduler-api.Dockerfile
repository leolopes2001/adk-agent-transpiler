# syntax=docker/dockerfile:1.7
# Scheduler API. Installs only the `api` extra; migrations run in the entrypoint.

FROM python:3.12.16-slim-trixie@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra api
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra api

FROM python:3.12.16-slim-trixie@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f
# Unprivileged user; no setuid/setgid binaries left to escalate from.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin app \
    && find / -xdev -perm /6000 -type f -exec chmod a-s {} +
COPY --from=builder /opt/venv /opt/venv
COPY --chmod=755 scripts/api-entrypoint.sh /usr/local/bin/api-entrypoint.sh
# The root filesystem is read-only at run time; HOME points at the /tmp tmpfs.
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
WORKDIR /app
# alembic reads alembic.ini from the working directory.
COPY alembic.ini ./
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/api-entrypoint.sh"]
