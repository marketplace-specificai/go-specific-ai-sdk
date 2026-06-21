#!/usr/bin/env bash
# Publish this go-sdk directory to the public repo and tag a new version.
# Usage: ./publish.sh <version>   e.g.  ./publish.sh v0.0.2
set -euo pipefail

REPO="https://github.com/marketplace-specificai/go-specific-ai-sdk.git"

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  echo "Usage: $0 <version>  (e.g. $0 v0.0.2)" >&2
  exit 1
fi
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Version must be semver like v0.0.2 (got: $VERSION)" >&2
  exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Make sure gh is authenticated and git can use its credentials.
gh auth status
gh auth setup-git

# 2. Push the directory contents to the public repo's main branch.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp -R "$SRC_DIR"/. "$TMP"/
cd "$TMP"
rm -rf .git
git init -q
git checkout -q -b main
git add -A
git commit -q -m "Publish $VERSION"
git remote add origin "$REPO"
git push -f -u origin main

# 3. Create and push the new version tag.
git tag "$VERSION"
git push origin "$VERSION"

echo "Published $VERSION to $REPO"
