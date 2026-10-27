#!/usr/bin/env bash
# Checks the isolation of the running Compose stack (`docker compose up -d --wait`):
#   - backend services cannot reach the internet, while the agent can;
#   - only the API port is published, and only on the loopback interface;
#   - project containers run as UID 10001, without capabilities, with
#     no-new-privileges and a read-only root filesystem.
# Usage: bash scripts/check_isolation.sh   (COMPOSE overrides the compose command)
set -uo pipefail

COMPOSE=${COMPOSE:-docker compose}
PROBE_URL=${PROBE_URL:-https://www.google.com}
SERVICES=(scheduler-api ocr-mcp rag-mcp)
failures=0

pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; failures=$((failures + 1)); }

compose() { $COMPOSE "$@"; }
exec_in() { compose exec -T "$@"; }
run_agent() { compose run --rm --no-deps -T --entrypoint "$1" agent "${@:2}"; }

reach="import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=5)
except Exception as exc:
    print(type(exc).__name__)
    sys.exit(1)"

container_of() { compose ps -q "$1"; }

echo "== network"
for svc in "${SERVICES[@]}"; do
    if exec_in "$svc" python -c "$reach" "$PROBE_URL" >/dev/null 2>&1; then
        fail "$svc reaches $PROBE_URL"
    else
        pass "$svc cannot reach the internet"
    fi
done
if exec_in postgres wget -q -T 5 -O /dev/null "$PROBE_URL" >/dev/null 2>&1; then
    fail "postgres reaches $PROBE_URL"
else
    pass "postgres cannot reach the internet"
fi
# Positive control: the probe works where egress is allowed.
if run_agent python -c "$reach" "$PROBE_URL" >/dev/null 2>&1; then
    pass "agent reaches $PROBE_URL (egress network)"
else
    fail "agent cannot reach $PROBE_URL"
fi
for svc in postgres "${SERVICES[@]}"; do
    networks=$(docker inspect -f '{{range $name, $_ := .NetworkSettings.Networks}}{{$name}} {{end}}' "$(container_of "$svc")")
    for network in $networks; do
        internal=$(docker network inspect -f '{{.Internal}}' "$network")
        if [[ $network == *_edge && $svc == scheduler-api ]] || [[ $internal == true ]]; then
            continue
        fi
        fail "$svc is attached to non-internal network $network"
    done
done

echo "== ports"
published=$(docker ps --filter "id=$(container_of scheduler-api)" --format '{{.Ports}}')
if [[ $published == "127.0.0.1:8000->8000/tcp" ]]; then
    pass "scheduler-api publishes only 127.0.0.1:8000"
else
    fail "scheduler-api publishes: $published"
fi
for pair in ocr-mcp:8101 rag-mcp:8102 postgres:5432; do
    svc=${pair%%:*}
    port=${pair##*:}
    if docker port "$(container_of "$svc")" "$port/tcp" >/dev/null 2>&1; then
        fail "$svc publishes port $port"
    else
        pass "$svc does not publish port $port"
    fi
done

echo "== user and privileges"
# Takes the /proc/<pid>/status of the service's main process.
check_process() {
    local name=$1 status=$2 uid
    uid=$(awk '/^Uid:/ {print $2}' <<<"$status")
    if [[ $uid == 10001 ]]; then pass "$name runs as UID 10001"; else fail "$name runs as UID $uid"; fi
    if grep -q '^CapEff:[[:space:]]*0000000000000000$' <<<"$status"; then
        pass "$name has no effective capabilities"
    else
        fail "$name has capabilities: $(grep '^CapEff' <<<"$status")"
    fi
    if grep -q '^NoNewPrivs:[[:space:]]*1$' <<<"$status"; then
        pass "$name has no-new-privileges"
    else
        fail "$name can gain privileges"
    fi
}
for svc in "${SERVICES[@]}"; do
    check_process "$svc" "$(exec_in "$svc" cat /proc/1/status)"
    if [[ $(exec_in "$svc" id -u) == 10001 ]]; then
        pass "$svc exec sessions run as UID 10001"
    else
        fail "$svc exec sessions do not run as UID 10001"
    fi
    if exec_in "$svc" touch /app/.write-probe >/dev/null 2>&1; then
        fail "$svc root filesystem is writable"
    else
        pass "$svc root filesystem is read-only"
    fi
done
check_process agent "$(run_agent cat /proc/self/status)"
if run_agent touch /app/.write-probe >/dev/null 2>&1; then
    fail "agent root filesystem is writable"
else
    pass "agent root filesystem is read-only"
fi
# The official Postgres image drops to its own unprivileged user after startup.
pg_uid=$(exec_in postgres sh -c 'stat -c %u /proc/1')
pg_user=$(exec_in postgres id -u postgres)
if [[ -n $pg_uid && $pg_uid == "$pg_user" && $pg_uid != 0 ]]; then
    pass "postgres server runs as the postgres user (UID $pg_uid)"
else
    fail "postgres server runs as UID $pg_uid"
fi

echo
if ((failures)); then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
