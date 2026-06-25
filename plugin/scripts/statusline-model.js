#!/usr/bin/env node
/**
 * Claude Code Status Line for claude-mem
 *
 * Reads Claude Code's stdin JSON (model, workspace, context_window),
 * enriches with claude-mem context stats (observations count, tokens),
 * outputs: "model | directory | Context: XX%"
 *
 * Configure in .claude/settings.local.json:
 * { "statusLine": { "type": "command", "command": "bash \"$HOME/.claude/statusline.sh\"" } }
 */

import { createRequire } from 'module';
import { stdin, stdout } from 'process';
import http from 'http';
import https from 'https';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const require = createRequire(import.meta.url);

// ============================================================================
// Read Claude Code stdin JSON
// ============================================================================

const chunks = [];
stdin.resume();
stdin.setEncoding('utf8');

stdin.on('data', (chunk) => chunks.push(chunk));

stdin.on('end', async () => {
  try {
    const claudeData = JSON.parse(chunks.join(''));
    const model = claudeData.model?.display_name || claudeData.model?.id || '?';
    let cwd = claudeData.workspace?.current_dir || '';

    // Convert /d/foo -> D:\foo on Windows
    cwd = cwd.replace(/^\/([a-z])\//, (_, drive) => drive.toUpperCase() + ':\\');

    // Get context usage from Claude Code if available
    const usedPercentage = claudeData.context_window?.used_percentage;

    // Build statusline parts
    const parts = [model, cwd];

    if (usedPercentage != null) {
      parts.push('Context: ' + Math.round(usedPercentage) + '%');
    }

    // Get claude-mem stats asynchronously
    try {
      const stats = await getClaudeMemStats(cwd);
      if (stats) {
        if (stats.totalObservations > 0) {
          parts.push('Obs: ' + stats.totalObservations);
        }
      }
    } catch {
      // Ignore stats errors
    }

    stdout.write(parts.join(' | '));
  } catch (e) {
    stdout.write('claude-mem');
  }
});

// ============================================================================
// Get claude-mem stats from worker API
// ============================================================================

async function getClaudeMemStats(cwd) {
  const project = path.basename(cwd) || 'default';
  const dataDir = getDataDir();
  const dbPath = path.join(dataDir, 'claude-mem.db');
  const settingsPath = path.join(dataDir, 'settings.json');

  // Get worker port
  let port = 37777;
  try {
    if (fs.existsSync(settingsPath)) {
      const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
      port = parseInt(settings.CLAUDE_MEM_WORKER_PORT) || 37777;
    }
  } catch { /* use default */ }

  // Try worker API first
  try {
    const stats = await fetchJson(`http://localhost:${port}/api/stats`, 1000);
    if (stats) {
      return stats;
    }
  } catch { /* try DB fallback */ }

  // Fallback: query SQLite directly
  if (fs.existsSync(dbPath)) {
    return await getStatsFromSqlite(dbPath, project);
  }

  return null;
}

// ============================================================================
// Get data directory
// ============================================================================

function getDataDir() {
  if (process.env.CLAUDE_MEM_DATA_DIR) {
    return process.env.CLAUDE_MEM_DATA_DIR;
  }
  return path.join(os.homedir(), '.claude-mem');
}

// ============================================================================
// HTTP fetch helper
// ============================================================================

function fetchJson(url, timeoutMs = 1000) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const client = urlObj.protocol === 'https:' ? https : http;

    const req = client.get(urlObj, { timeout: timeoutMs }, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(e);
        }
      });
    });

    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('timeout'));
    });
  });
}

// ============================================================================
// SQLite direct query (cross-platform)
// ============================================================================

async function getStatsFromSqlite(dbPath, project) {
  // Use sqlite3 CLI if available, otherwise try better-sqlite3
  try {
    const stats = await querySqlite(dbPath,
      `SELECT COUNT(*) as totalObservations FROM observations WHERE project = ?`,
      [project]
    );
    return stats;
  } catch (e) {
    return null;
  }
}

function querySqlite(dbPath, sql, params = []) {
  return new Promise((resolve, reject) => {
    // Build command with proper escaping
    const escapedParams = params.map(p => {
      if (p === undefined || p === null) return 'NULL';
      const str = String(p);
      return `'${str.replace(/'/g, "''")}'`;
    });

    const fullSql = sql.replace(/\?/g, () => escapedParams.shift());

    // Try sqlite3 CLI first (cross-platform)
    const child = spawn('sqlite3', [dbPath, '-json', fullSql], {
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdoutData = '';
    let stderrData = '';

    child.stdout.on('data', (data) => stdoutData += data.toString());
    child.stderr.on('data', (data) => stderrData += data.toString());

    child.on('close', (code) => {
      if (code === 0 && stdoutData.trim()) {
        try {
          const result = JSON.parse(stdoutData.trim());
          resolve(result[0] || {});
        } catch (e) {
          reject(e);
        }
      } else {
        reject(new Error(stderrData || `sqlite3 exited with code ${code}`));
      }
    });

    child.on('error', (e) => {
      reject(e);
    });
  });
}
