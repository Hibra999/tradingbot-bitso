#!/usr/bin/env bash
set -euo pipefail

readonly PUFFERLIB_VERSION="3.0.0"
readonly PUFFERLIB_SHA256="7df3a3e3f5f894d78d2a1f5374097890aec01473183e748abefe4f3faa10eaa9"
readonly PUFFERLIB_URL="https://files.pythonhosted.org/packages/7c/e1/5292f9b69c6263707b40ba04a87e6b9bcc177281d31092f77afd90c412f1/pufferlib-3.0.0.tar.gz"
readonly REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TEMP_DIRECTORY="$(mktemp -d)"
readonly ARCHIVE_PATH="$TEMP_DIRECTORY/pufferlib-$PUFFERLIB_VERSION.tar.gz"
readonly SOURCE_DIRECTORY="$TEMP_DIRECTORY/pufferlib-$PUFFERLIB_VERSION"

trap 'rm -rf -- "$TEMP_DIRECTORY"' EXIT

curl --fail --location --silent --show-error "$PUFFERLIB_URL" --output "$ARCHIVE_PATH"
printf '%s  %s\n' "$PUFFERLIB_SHA256" "$ARCHIVE_PATH" | sha256sum --check --status
tar -xzf "$ARCHIVE_PATH" -C "$TEMP_DIRECTORY"

(
    cd "$SOURCE_DIRECTORY"
    git apply "$REPOSITORY_ROOT/patches/pufferlib-3.0.0-no-ocean.patch"
)

NO_OCEAN=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
    python -m pip install --no-build-isolation --no-deps "$SOURCE_DIRECTORY"
