/**
 * NeuroScan AI — Render Build Script
 * 1. Copies ml_artifacts metrics JSON into ../ml/artifacts/ (overwrite Render cache)
 * 2. Copies ALL frontend files into server/public/ so Render serves fresh versions
 *    (fixes stale-cache mojibake issue where Render serves old HTML)
 */
const fs   = require('fs');
const path = require('path');

const SERVER_DIR = __dirname;
const ROOT_DIR   = path.join(__dirname, '..');

// ── 1. Copy metrics JSON ──────────────────────────────────────────
const mlSrc = path.join(SERVER_DIR, 'ml_artifacts');
const mlDst = path.join(ROOT_DIR, 'ml', 'artifacts');
if (fs.existsSync(mlSrc)) {
  if (!fs.existsSync(mlDst)) fs.mkdirSync(mlDst, { recursive: true });
  fs.readdirSync(mlSrc).forEach(f => {
    fs.copyFileSync(path.join(mlSrc, f), path.join(mlDst, f));
    console.log('Copied', f, '→ ml/artifacts/');
  });
}

// ── 2. Copy frontend into server/public/ ─────────────────────────
// This ensures Render serves fresh HTML/JS/CSS regardless of its disk cache
const publicDir = path.join(SERVER_DIR, 'public');
if (!fs.existsSync(publicDir)) fs.mkdirSync(publicDir, { recursive: true });

function copyDir(src, dst) {
  if (!fs.existsSync(src)) return;
  if (!fs.existsSync(dst)) fs.mkdirSync(dst, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const dstPath = path.join(dst, entry.name);
    if (entry.isDirectory()) {
      // Skip node_modules, .git, server itself, __pycache__
      if (['node_modules','.git','server','__pycache__','.kiro'].includes(entry.name)) continue;
      copyDir(srcPath, dstPath);
    } else {
      fs.copyFileSync(srcPath, dstPath);
    }
  }
}

// Copy all frontend directories
['pages','css','js','data','ml'].forEach(dir => {
  const src = path.join(ROOT_DIR, dir);
  const dst = path.join(publicDir, dir);
  copyDir(src, dst);
  if (fs.existsSync(src)) console.log('Copied', dir, '→ server/public/' + dir + '/');
});

// Copy root-level files
['index.html', 'metadata.json'].forEach(f => {
  const src = path.join(ROOT_DIR, f);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, path.join(publicDir, f));
    console.log('Copied', f, '→ server/public/');
  }
});

console.log('Build complete. server/public/ is ready.');
