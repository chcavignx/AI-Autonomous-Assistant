#!/bin/bash
arecord -f cd -d 5 test.wav
#!/bin/bash
set -e  # Exit on any error

# Clean up test file on exit
cleanup() {
  rm -f test.wav
}
trap cleanup EXIT

# Verify required commands exist
command -v arecord >/dev/null 2>&1 || { echo "arecord not found"; exit 1; }
command -v aplay >/dev/null 2>&1 || { echo "aplay not found"; exit 1; }

arecord -f cd -d 5 test.wav
aplay test.wav
