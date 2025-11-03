## TiRex Docs

The product documentation and auto-generated Python API reference live in the `docs/` directory. We use Docusaurus for the site shell and Sphinx ➜ Markdown for the API pages.

---

### 1. Prerequisites

```bash
# (optional) create an isolated env for Sphinx + TiRex imports
python -m venv .venv && source .venv/bin/activate

# install the project so autodoc can import tirex
pip install -e .
```

> The Sphinx generator will install its own dependencies (sphinx, myst, etc.) automatically, so no extra requirements file is needed.

---

### 2. Install JavaScript deps

```bash
npm --prefix docs install
```

Run this once (or whenever `package.json` changes).

---

### 3. Regenerate the API docs (optional but recommended)

```bash
# produces Markdown under docs/content/api/generated
npm --prefix docs run generate:api
```

This wraps `docs/scripts/generate_api_docs_sphinx.sh`, which:

- builds fresh Sphinx stubs for the Python package
- renders them to Markdown
- escapes MDX-sensitive characters
- refreshes Docusaurus category metadata

Whenever you change docstrings or add modules, re-run this command. If Docusaurus starts complaining about stale caches, clear `docs/.docusaurus` and `docs/node_modules/.cache`.

---

### 4. Run the docs locally

```bash
npm --prefix docs run start
```

This launches the dev server on `http://localhost:3000/` with hot reload.

---

### 5. Production build preview

```bash
npm --prefix docs run build
npm --prefix docs run serve   # optional: serve the static build locally
```

The build step runs during CI/deployment. Failures usually indicate missing redirects, broken links, or unescaped characters in the generated API Markdown.

---

### Useful scripts & files

- `docs/scripts/generate_api_docs_sphinx.sh` – orchestrates the Sphinx ➜ Markdown pipeline
- `docs/content/api/index.mdx` – overview page for the API reference
- `docs/docusaurus.config.js` – global site configuration
- `docs/sidebars.js` – sidebar structure
