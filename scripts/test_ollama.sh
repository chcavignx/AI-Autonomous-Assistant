#!/usr/bin/env bash

# Re-execute with bash if run with a different shell (e.g. sh)
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -Eeuo pipefail

# Determine script directory and repository root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

# Logging functions
log_info() {
    printf "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: %s\n" "$*" >&2
}

log_error() {
    printf "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: %s\n" "$*" >&2
}

# 1. Parse configuration using python in uv environment
log_info "Reading configuration from config.yaml..."
LLM_URL=$(uv run python3 -c "import yaml; data = yaml.safe_load(open('$REPO_ROOT/config.yaml')); print(data.get('llm', {}).get('url', 'http://127.0.0.1:11434/api/generate'))")
LLM_MODEL=$(uv run python3 -c "import yaml; data = yaml.safe_load(open('$REPO_ROOT/config.yaml')); print(data.get('llm', {}).get('model', 'llama3.2:latest'))")
LLM_TIMEOUT=$(uv run python3 -c "import yaml; data = yaml.safe_load(open('$REPO_ROOT/config.yaml')); print(int(data.get('llm', {}).get('timeout', 10)))")
TIMEOUT="${OLLAMA_TIMEOUT:-$LLM_TIMEOUT}"

# Resolve tags URL from generate URL
BASE_URL="${LLM_URL%/api/generate}"
TAGS_URL="$BASE_URL/api/tags"

log_info "Configuration loaded:"
log_info "  Ollama Base URL : $BASE_URL"
log_info "  Generate URL    : $LLM_URL"
log_info "  Target Model    : $LLM_MODEL"
log_info "  Timeout         : ${TIMEOUT}s"

# 2. Check if Ollama is running
log_info "Checking if Ollama service is reachable..."
if ! curl -s -f -m "$TIMEOUT" "$BASE_URL" > /dev/null; then
    log_error "Ollama service is NOT reachable at $BASE_URL."
    log_error "Please make sure the Ollama server is running (e.g., run: ollama serve)."
    exit 1
fi
log_info "✓ Ollama service is running and responsive."

# 3. Check if target model is pulled
log_info "Checking if model '$LLM_MODEL' is pulled..."
TAGS_RESPONSE=$(curl -s -f -m "$TIMEOUT" "$TAGS_URL" || echo "")
if [[ -z "$TAGS_RESPONSE" ]]; then
    log_info "Could not retrieve installed models list from $TAGS_URL."
else
    # Check if LLM_MODEL exists in tags response
    if ! echo "$TAGS_RESPONSE" | grep -Fq "\"$LLM_MODEL\""; then
        log_error "Model '$LLM_MODEL' is NOT pulled on this Ollama server."
        log_error "Please run: ollama pull $LLM_MODEL"
        exit 1
    fi
    log_info "✓ Model '$LLM_MODEL' is available."
fi

# 4. Measure time to answer
TEST_PROMPT="State in one short sentence what is the capital of France."
log_info "Sending query to Ollama (Prompt: '$TEST_PROMPT')..."

# Build payload
PAYLOAD=$(cat <<EOF
{
  "model": "$LLM_MODEL",
  "prompt": "$TEST_PROMPT",
  "stream": false
}
EOF
)

START_TIME=$(date +%s.%N)

# Perform request and capture response
HTTP_RESPONSE=$(curl -s -w "\n%{http_code}" -m "$TIMEOUT" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "$LLM_URL" || true)

END_TIME=$(date +%s.%N)

# Separate body and HTTP code
HTTP_CODE=$(echo "$HTTP_RESPONSE" | tail -n 1)
RESPONSE_BODY=$(echo "$HTTP_RESPONSE" | sed '$d')

if [[ "$HTTP_CODE" != "200" ]]; then
    log_error "Request failed with HTTP status: $HTTP_CODE"
    log_error "Response: $RESPONSE_BODY"
    exit 1
fi

# Calculate duration
if [[ "$START_TIME" == *.* && "$END_TIME" == *.* ]]; then
    ELAPSED_TIME=$(uv run python3 -c "print(f'{float($END_TIME) - float($START_TIME):.3f}')")
else
    ELAPSED_TIME=$(( $(date +%s) - ${START_TIME%.*} ))
fi

# Parse response text using python json parser
LLM_ANSWER=$(echo "$RESPONSE_BODY" | uv run python3 -c "import sys, json; print(json.load(sys.stdin).get('response', '').strip())" || echo "Error parsing JSON response")

log_info "✓ Successful query!"
echo "============================================================"
echo "OLLAMA TEST RESULTS:"
echo "  Response Time : $ELAPSED_TIME seconds"
echo "  Model Answer  : $LLM_ANSWER"
echo "============================================================"
