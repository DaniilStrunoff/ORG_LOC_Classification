#!/bin/sh
set -eu

/run.sh & pid=$!

BASE_URL="http://127.0.0.1:3000"
AUTH="${GF_SECURITY_ADMIN_USER}:${GF_SECURITY_ADMIN_PASSWORD}"

if ! command -v curl >/dev/null 2>&1; then
  apk add --no-cache curl >/dev/null 2>&1 || true
fi

for i in $(seq 1 60); do
  if curl -sf "${BASE_URL}/api/health" >/dev/null 2>&1; then break; fi
  sleep 1
done

EXISTS_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  -u "${AUTH}" \
  "${BASE_URL}/api/users/lookup?loginOrEmail=${GRAFANA_TECH_USER_LOGIN}" || true)

if [ "${EXISTS_CODE}" != "200" ]; then
  BODY=$(printf '{"name":"%s","login":"%s","password":"%s"}' \
        "${GRAFANA_TECH_USER_NAME}" "${GRAFANA_TECH_USER_LOGIN}" "${GRAFANA_TECH_USER_PASSWORD}")
  curl -sS -u "${AUTH}" -H "Content-Type: application/json" -d "${BODY}" \
       "${BASE_URL}/api/admin/users" >/dev/null || true
fi

USER_ID=$(curl -s -u "${AUTH}" \
  "${BASE_URL}/api/users/lookup?loginOrEmail=${GRAFANA_TECH_USER_LOGIN}" \
  | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*\([0-9]\+\).*/\1/p')

if [ -n "${USER_ID:-}" ]; then
  ROLE="${GRAFANA_TECH_USER_ROLE:-Editor}"
  SET_CODE=$(curl -s -o /dev/null -w "%{http_code}" -u "${AUTH}" \
    -H "Content-Type: application/json" -X PATCH -d "{\"role\":\"${ROLE}\"}" \
    "${BASE_URL}/api/orgs/1/users/${USER_ID}" || true)
  if [ "${SET_CODE}" = "404" ]; then
    ADD_BODY=$(printf '{"loginOrEmail":"%s","role":"%s"}' "${GRAFANA_TECH_USER_LOGIN}" "${ROLE}")
    curl -sS -u "${AUTH}" -H "Content-Type: application/json" -d "${ADD_BODY}" \
         "${BASE_URL}/api/orgs/1/users" >/dev/null || true
  fi
fi

wait $pid
