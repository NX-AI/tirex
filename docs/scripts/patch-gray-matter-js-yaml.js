#!/usr/bin/env node

/**
 * gray-matter depends on js-yaml 3.x APIs (safeLoad/safeDump). We override js-yaml
 * to 4.x for security fixes, so we add compatibility shims to keep parsing working.
 * This script is idempotent and runs automatically after npm install.
 */

const fs = require('fs');
const path = require('path');

const enginePath = path.join(__dirname, '..', 'node_modules', 'gray-matter', 'lib', 'engines.js');

function patchGrayMatter() {
  if (!fs.existsSync(enginePath)) {
    console.warn('gray-matter not found, skipping js-yaml compatibility patch');
    return;
  }

  const original = fs.readFileSync(enginePath, 'utf8');

  if (original.includes('yaml.load || yaml.safeLoad')) {
    console.log('gray-matter already patched for js-yaml v4 compatibility');
    return;
  }

  const withShim = original
    .replace(
      "const yaml = require('js-yaml');",
      "const yaml = require('js-yaml');\nconst yamlLoad = yaml.load || yaml.safeLoad;\nconst yamlDump = yaml.dump || yaml.safeDump;"
    )
    .replace('parse: yaml.safeLoad.bind(yaml),', 'parse: yamlLoad.bind(yaml),')
    .replace('stringify: yaml.safeDump.bind(yaml)', 'stringify: yamlDump.bind(yaml)');

  if (withShim === original) {
    console.warn('gray-matter engines.js format unexpected; no changes applied');
    return;
  }

  fs.writeFileSync(enginePath, withShim);
  console.log('Patched gray-matter to use js-yaml load/dump (v4 compatible)');
}

patchGrayMatter();
