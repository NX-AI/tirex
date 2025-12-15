#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Ensure package + doc deps present (idempotent)
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
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

    # Add stable heading anchors for Docusaurus (needed for intra-page links)
    if md_path.name == "tirex.classification.md":
        text = re.sub(
            r"^(### \*class\* tirex\.models\.classification\.TirexLinearClassifier[^\n]*)(\n)",
            r"\1 {#tirex.models.classification.TirexLinearClassifier}\2",
            text,
            flags=re.M,
        )
        text = re.sub(
            r"^(### \*class\* tirex\.models\.classification\.TirexRFClassifier[^\n]*)(\n)",
            r"\1 {#tirex.models.classification.TirexRFClassifier}\2",
            text,
            flags=re.M,
        )
        text = re.sub(
            r"^(### \*class\* tirex\.models\.classification\.TirexGBMClassifier[^\n]*)(\n)",
            r"\1 {#tirex.models.classification.TirexGBMClassifier}\2",
            text,
            flags=re.M,
        )

    if md_path.name == "tirex.regression.md":
        text = re.sub(
            r"^(### \*class\* tirex\.models\.regression\.TirexLinearRegressor[^\n]*)(\n)",
            r"\1 {#tirex.models.regression.TirexLinearRegressor}\2",
            text,
            flags=re.M,
        )
        text = re.sub(
            r"^(### \*class\* tirex\.models\.regression\.TirexRFRegressor[^\n]*)(\n)",
            r"\1 {#tirex.models.regression.TirexRFRegressor}\2",
            text,
            flags=re.M,
        )
        text = re.sub(
            r"^(### \*class\* tirex\.models\.regression\.TirexGBMRegressor[^\n]*)(\n)",
            r"\1 {#tirex.models.regression.TirexGBMRegressor}\2",
            text,
            flags=re.M,
        )

    # Preserve explicit anchors while escaping braces for MDX
    anchors: dict[str, str] = {}

    def _stash_anchor(match: re.Match[str]) -> str:
        key = f"__ANCHOR_{len(anchors)}__"
        anchors[key] = match.group(0)
        return key

    text = re.sub(r"\{#[-_.A-Za-z0-9]+\}", _stash_anchor, text)

    text = re.sub(r"<class ([^>]+)>", lambda m: f"&lt;class {m.group(1)}&gt;", text)
    text = text.replace("{", "&#123;").replace("}", "&#125;")
    text = re.sub(r"<(?=[^a-zA-Z!/])", "&lt;", text)

    for placeholder, anchor in anchors.items():
        text = text.replace(placeholder, anchor)

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
