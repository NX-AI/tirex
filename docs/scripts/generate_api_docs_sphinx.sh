#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Ensure package + doc deps present (idempotent)
python -m pip install -e .
python -m pip install \
  sphinx sphinx-markdown-builder sphinx-autobuild myst-parser sphinxcontrib-mermaid \
  sphinx-autodoc-typehints sphinx-copybutton sphinx-design

TARGET_MD_DIR="docs/content/api/generated"

# Clean old generated API markdown (reST sources are curated manually)
rm -rf "$TARGET_MD_DIR"
mkdir -p "$TARGET_MD_DIR"

# Build to Markdown; builder name is "markdown"
sphinx-build -b markdown docs/sphinx/source "$TARGET_MD_DIR"

# Escape characters that break MDX parsing in Docusaurus
python - "$TARGET_MD_DIR" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

for md_path in root.glob("*.md"):
    text = md_path.read_text()
    text = re.sub(r"<class ([^>]+)>", lambda m: f"&lt;class {m.group(1)}&gt;", text)
    text = text.replace("{", "&#123;").replace("}", "&#125;")
    text = re.sub(r"<(?=[^a-zA-Z!/])", "&lt;", text)
    md_path.write_text(text)
PY

cat > "$TARGET_MD_DIR/_category_.json" <<'JSON'
{
  "label": "Python Modules",
  "collapsed": false,
  "link": {
    "type": "doc",
    "id": "api/generated/index"
  },
  "className": "icon-python"
}
JSON

echo "Sphinx → Markdown complete: $TARGET_MD_DIR"
