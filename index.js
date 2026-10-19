#!/usr/bin/env node
import OpenAI from "openai";
import readlineSync from "readline-sync";
import { exec, execFile } from "child_process";
import { promisify } from "util";
import fs from "fs";
import path from "path";
import os from "os";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

// ============================================================
// CONFIG
// ============================================================

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Layout:
//   ai-agent/
//   ├── index.js        (this file: the agent)
//   ├── package.json
//   ├── node_modules/
//   ├── .env            (API key, kept outside the AI's reach)
//   └── project/        (the ONLY folder the AI can read/edit/add to; git lives here)
const PROJECT_DIR = path.join(__dirname, "project");
fs.mkdirSync(PROJECT_DIR, { recursive: true });

dotenv.config({ path: path.join(__dirname, ".env") });
dotenv.config({ path: path.join(PROJECT_DIR, ".env") }); // also works if you keep .env inside project/

const CWD = PROJECT_DIR; // the ONLY directory the agent may touch
const PLATFORM = os.platform();
let MODEL = process.env.MODEL || ""; // optional override in .env, e.g. MODEL=openai/gpt-oss-120b

// Preferred models (best first). Only used if your key can actually access them.
const PREFERRED_MODELS = [
  "openai/gpt-oss-120b",
  "llama-3.3-70b-versatile",
  "moonshotai/kimi-k2-instruct",
  "qwen/qwen3-32b",
  "meta-llama/llama-4-scout-17b-16e-instruct",
  "openai/gpt-oss-20b",
  "llama-3.1-8b-instant",
];
const NON_CHAT_MODEL = /whisper|tts|guard|embed|orpheus|playai|safeguard|transcri|speech/i;
const API_KEY = process.env.GROQ_API_KEY || process.env.apikey;

const MAX_ITERATIONS = 15;
const MAX_CONTEXT_CHARS = 24000; // total file content sent each prompt (free tier has token limits)
const MAX_FILE_CHARS = 6000; // per-file cap inside the snapshot
const MAX_TOOL_OUTPUT = 4000; // cap on any tool result sent back to the model
const COMMAND_TIMEOUT_MS = 120000;

const IGNORED_DIRS = new Set([
  ".git", "node_modules", "dist", "build", ".next", ".nuxt", "out",
  "coverage", "__pycache__", ".venv", "venv", ".idea", ".vscode", ".cache",
]);
const IGNORED_FILES = new Set([
  ".env", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", ".DS_Store",
]);
const BINARY_EXT = new Set([
  ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz",
  ".mp3", ".mp4", ".woff", ".woff2", ".ttf", ".eot", ".exe", ".dll", ".so",
]);

const run = promisify(exec);
const runFile = promisify(execFile);

if (!API_KEY) {
  console.error("Missing API key. Put GROQ_API_KEY=... in a .env file.");
  process.exit(1);
}

const client = new OpenAI({
  apiKey: API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});

// ============================================================
// GIT HELPERS
// ============================================================

async function git(args) {
  const { stdout } = await runFile("git", args, {
    cwd: CWD,
    maxBuffer: 10 * 1024 * 1024,
  });
  return stdout.trim();
}

async function gitConfigured() {
  try {
    return (await git(["config", "user.name"])) !== "";
  } catch {
    return false;
  }
}

async function gitCommit(message) {
  const identity = (await gitConfigured())
    ? []
    : ["-c", "user.name=AI Assistant", "-c", "user.email=ai@local"];
  await git(["add", "-A"]);
  await git([...identity, "commit", "-m", message, "--allow-empty-message"]);
}

async function isDirty() {
  return (await git(["status", "--porcelain"])) !== "";
}

async function hasCommits() {
  try {
    await git(["rev-parse", "HEAD"]);
    return true;
  } catch {
    return false;
  }
}

function ensureGitignore() {
  const file = path.join(CWD, ".gitignore");
  const current = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
  const lines = current.split(/\r?\n/);
  const missing = [".env", "node_modules/"].filter((l) => !lines.includes(l));
  if (missing.length) {
    const prefix = current && !current.endsWith("\n") ? "\n" : "";
    fs.appendFileSync(file, prefix + missing.join("\n") + "\n");
  }
}

async function setupGit() {
  if (!fs.existsSync(path.join(CWD, ".git"))) {
    await git(["init"]);
    console.log("📁 Initialized a new git repository.");
  }
  ensureGitignore();
  // baseline snapshot so every later prompt commit is a clean diff
  if (!(await hasCommits()) && (await isDirty())) {
    await gitCommit("0th snapshot 'initial project state'");
    console.log("📸 Saved initial project state as baseline commit.");
  }
}

function ordinal(n) {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

// Returns [{ hash, subject, when, promptNo|null }] oldest -> newest
async function listCommits() {
  if (!(await hasCommits())) return [];
  const out = await git(["log", "--reverse", "--pretty=format:%h\t%s\t%cr"]);
  return out
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      const [hash, subject, when] = line.split("\t");
      const m = subject.match(/^(\d+)(?:st|nd|rd|th) prompt /);
      return { hash, subject, when, promptNo: m ? Number(m[1]) : null };
    });
}

async function nextPromptNumber() {
  const commits = await listCommits();
  const nums = commits.map((c) => c.promptNo).filter((n) => n !== null);
  return (nums.length ? Math.max(...nums) : 0) + 1;
}

// ============================================================
// PROJECT SNAPSHOT (sent to the AI on EVERY prompt)
// ============================================================

function collectFiles(dir = CWD, out = []) {
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const entry of entries) {
    const abs = path.join(dir, entry.name);
    if (abs === __filename) continue; // never show the agent its own source
    if (entry.isDirectory()) {
      if (!IGNORED_DIRS.has(entry.name)) collectFiles(abs, out);
    } else if (entry.isFile()) {
      if (IGNORED_FILES.has(entry.name)) continue;
      if (BINARY_EXT.has(path.extname(entry.name).toLowerCase())) continue;
      out.push({
        rel: path.relative(CWD, abs).split(path.sep).join("/"),
        abs,
        size: fs.statSync(abs).size,
      });
    }
  }
  return out;
}

function buildSnapshot() {
  const files = collectFiles().sort((a, b) => a.rel.localeCompare(b.rel));

  if (files.length === 0) {
    return `PROJECT DIRECTORY: ${CWD}\nThe project is EMPTY. There are no files yet.`;
  }

  // fill the content budget with the smallest files first, so more files fit
  const bySize = [...files].sort((a, b) => a.size - b.size);
  let budget = MAX_CONTEXT_CHARS;
  const contents = new Map();
  for (const f of bySize) {
    let text = fs.readFileSync(f.abs, "utf8");
    if (text.includes("\u0000")) {
      contents.set(f.rel, "(binary file, content omitted)");
      continue;
    }
    if (text.length > MAX_FILE_CHARS) {
      text = text.slice(0, MAX_FILE_CHARS) + "\n...[truncated, use read_file for the rest]";
    }
    if (text.length > budget) {
      contents.set(f.rel, "(content omitted to save context, use read_file)");
      continue;
    }
    budget -= text.length;
    contents.set(f.rel, text);
  }

  const tree = files.map((f) => `- ${f.rel} (${f.size} bytes)`).join("\n");
  const body = files
    .map((f) => `===== FILE: ${f.rel} =====\n${contents.get(f.rel)}`)
    .join("\n\n");

  return `PROJECT DIRECTORY: ${CWD}\n\nFILE STRUCTURE:\n${tree}\n\nFILE CONTENTS:\n${body}`;
}

// ============================================================
// SANDBOXED TOOLS
// ============================================================

function safePath(p) {
  if (typeof p !== "string" || !p.trim()) throw new Error("Path is required.");
  const abs = path.resolve(CWD, p);
  const rel = path.relative(CWD, abs);
  if (rel === "" || rel.startsWith("..") || path.isAbsolute(rel)) {
    throw new Error(`Access denied: "${p}" is outside the project directory.`);
  }
  const first = rel.split(path.sep)[0];
  if (first === ".git" || rel === ".env") {
    throw new Error(`Access denied: "${p}" is protected.`);
  }
  if (abs === __filename) throw new Error("Access denied: agent source file.");
  return abs;
}

const BLOCKED_COMMANDS = [
  /\brm\s+(-[a-z]*\s+)*(\/|~|\*|\.\.)/i,
  /\bsudo\b/i,
  /\b(shutdown|reboot|mkfs|format|diskpart)\b/i,
  /\bdel\s+\/[sq]/i,
  /\brmdir\s+\/s/i,
  /\bRemove-Item\b.*-Recurse/i,
  /\bgit\s+(commit|reset|checkout|switch|clean|push|rebase|branch\s+-D|stash)\b/i,
  /\b(curl|wget|iwr|Invoke-WebRequest)\b.*\|\s*(sh|bash|iex|powershell)/i,
  /\bcd\s+(\.\.|\/|~|[a-zA-Z]:)/i,
  /\.\.[\\/]/,
];

const tools = {
  async read_file({ path: p }) {
    const text = fs.readFileSync(safePath(p), "utf8");
    return text.length > 12000 ? text.slice(0, 12000) + "\n...[truncated]" : text;
  },

  async write_file({ path: p, content }) {
    const abs = safePath(p);
    fs.mkdirSync(path.dirname(abs), { recursive: true });
    fs.writeFileSync(abs, content ?? "", "utf8");
    return `Wrote ${p} (${(content ?? "").length} chars).`;
  },

  async replace_in_file({ path: p, old_str, new_str }) {
    const abs = safePath(p);
    const text = fs.readFileSync(abs, "utf8");
    const count = text.split(old_str).length - 1;
    if (count === 0) throw new Error("old_str not found in file.");
    if (count > 1) throw new Error(`old_str matches ${count} places; make it unique.`);
    fs.writeFileSync(abs, text.replace(old_str, () => new_str ?? ""), "utf8");
    return `Edited ${p}.`;
  },

  async delete_file({ path: p }) {
    const abs = safePath(p);
    fs.rmSync(abs, { recursive: true, force: true });
    return `Deleted ${p}.`;
  },

  async run_command({ command }) {
    if (typeof command !== "string" || !command.trim()) throw new Error("Command is required.");
    for (const rule of BLOCKED_COMMANDS) {
      if (rule.test(command)) throw new Error(`Blocked command: ${command}`);
    }
    try {
      const { stdout, stderr } = await run(command, {
        cwd: CWD,
        timeout: COMMAND_TIMEOUT_MS,
        windowsHide: true,
        maxBuffer: 10 * 1024 * 1024,
      });
      return (
        (stdout ? `STDOUT:\n${stdout}\n` : "") +
        (stderr ? `STDERR:\n${stderr}\n` : "") ||
        "Command executed successfully with no output."
      );
    } catch (err) {
      return (
        `Command failed (exit ${err.code ?? "?"}${err.killed ? ", timed out" : ""}).\n` +
        (err.stdout ? `STDOUT:\n${err.stdout}\n` : "") +
        (err.stderr ? `STDERR:\n${err.stderr}\n` : "") +
        (!err.stdout && !err.stderr ? err.message : "")
      );
    }
  },
};

const toolDefs = [
  {
    name: "read_file",
    description: "Read a file inside the project (use for files omitted/truncated in the snapshot).",
    parameters: {
      type: "object",
      properties: { path: { type: "string", description: "Relative path" } },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Create a file or completely overwrite it with full content. Parent folders are created automatically.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Relative path" },
        content: { type: "string", description: "Complete file content" },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "replace_in_file",
    description: "Replace one exact, unique piece of text in an existing file. Prefer this for small edits.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string" },
        old_str: { type: "string", description: "Exact text to replace (must appear once)" },
        new_str: { type: "string" },
      },
      required: ["path", "old_str", "new_str"],
    },
  },
  {
    name: "delete_file",
    description: "Delete a file or folder. Only when the user explicitly asks or the change requires it.",
    parameters: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  },
  {
    name: "run_command",
    description: "Run one shell command inside the project directory (npm install, node file.js, tests...). Never start long-running servers.",
    parameters: {
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
    },
  },
].map((fn) => ({ type: "function", function: fn }));

// ============================================================
// SYSTEM PROMPT
// ============================================================

const SYSTEM_PROMPT = `
You are an expert AI coding assistant working in a single project folder.
Operating system: ${PLATFORM}

You are STATELESS. You have no memory of earlier requests. Every request
comes with a fresh snapshot of the project (file structure + file contents).
The snapshot is the only source of truth about the current state.

RULES:
1. If the user only asks a question or wants an explanation, answer it directly
   in text. Do NOT call any tool that changes files for a pure question.
2. If the user wants something built, changed, or fixed, do it with the tools.
   Never just paste code into the chat when the task is to modify the project.
3. If the project is EMPTY, scaffold what the request needs from scratch
   (for a basic website: index.html, style.css, script.js).
4. Use only relative paths inside the project. Never touch anything outside it.
5. Prefer replace_in_file for small edits; use write_file for new files or rewrites.
   Always write complete, working code, never placeholders.
6. Do not run git commit/reset/checkout. Version control is handled for you.
7. Never start dev servers or watchers (npm start, npm run dev, nodemon...),
   they block forever. Short commands like npm install or node script.js are fine.
8. If a command fails, read the error and fix the cause. Never repeat the same failing command.
9. Keep code clean, responsive (for web), and well organised.
10. When finished, reply with a short summary of what you did.
`;

// ============================================================
// MODEL DISCOVERY (asks Groq which models your key can use)
// ============================================================

async function resolveModel() {
  let available = [];
  try {
    const list = await client.models.list();
    available = list.data
      .map((m) => m.id)
      .filter((id) => !NON_CHAT_MODEL.test(id))
      .sort();
  } catch (err) {
    console.error(`❌ Could not fetch model list: ${err.message}`);
    console.error("Check your GROQ_API_KEY and internet connection.");
    process.exit(1);
  }

  if (!available.length) {
    console.error("❌ No chat models available for this API key.");
    process.exit(1);
  }

  if (MODEL && available.includes(MODEL)) return;
  if (MODEL) console.log(`⚠️  MODEL "${MODEL}" is not available for your key. Choosing another...`);

  const preferred = PREFERRED_MODELS.find((m) => available.includes(m));
  if (preferred) {
    MODEL = preferred;
    return;
  }

  console.log("\nAvailable models:");
  available.forEach((id, i) => console.log(`  ${i + 1}. ${id}`));
  const idx = readlineSync.questionInt("\nPick a model number: ", { limitMessage: "Enter a valid number." });
  MODEL = available[Math.min(Math.max(idx, 1), available.length) - 1];
}

// ============================================================
// AGENT (one prompt = one independent run, no chat history)
// ============================================================

async function callModel(messages) {
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      return await client.chat.completions.create({
        model: MODEL,
        messages,
        tools: toolDefs,
        tool_choice: "auto",
        temperature: 0.2,
      });
    } catch (err) {
      const retriable = err.status === 429 || err.status >= 500;
      if (!retriable || attempt === 3) throw err;
      const wait = attempt * 8;
      console.log(`⏳ API busy (${err.status}). Retrying in ${wait}s...`);
      await new Promise((r) => setTimeout(r, wait * 1000));
    }
  }
}

async function runAgent(userPrompt) {
  await setupGit();
  const n = await nextPromptNumber();

  // Fresh context every time: system config + project snapshot + current prompt only.
  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    {
      role: "user",
      content: `${buildSnapshot()}\n\n===== USER REQUEST =====\n${userPrompt}`,
    },
  ];

  let finished = false;

  for (let i = 1; i <= MAX_ITERATIONS && !finished; i++) {
    console.log(`\n🤖 Step ${i}/${MAX_ITERATIONS}`);
    const response = await callModel(messages);
    const message = response.choices[0].message;

    if (message.tool_calls?.length) {
      messages.push(message);
      for (const call of message.tool_calls) {
        const name = call.function.name;
        let result;
        try {
          const args = JSON.parse(call.function.arguments || "{}");
          if (!tools[name]) throw new Error(`Unknown tool: ${name}`);
          console.log(`🛠️  ${name}`, args.path || args.command || "");
          result = await tools[name](args);
        } catch (err) {
          result = `ERROR: ${err.message}`;
          console.log(`⚠️  ${result}`);
        }
        messages.push({
          role: "tool",
          tool_call_id: call.id,
          content: String(result).slice(0, MAX_TOOL_OUTPUT),
        });
      }
      continue;
    }

    console.log("\n🤖 Assistant:\n" + (message.content || "(no response)"));
    finished = true;
  }

  if (!finished) console.log("\n⚠️ Maximum agent iterations reached.");

  // Commit only if the project actually changed (questions leave no commit).
  if (await isDirty()) {
    const status = await git(["status", "--short"]);
    const clean = userPrompt.replace(/\s+/g, " ").replace(/'/g, "’").slice(0, 200);
    await gitCommit(`${ordinal(n)} prompt '${clean}'`);
    console.log(`\n✅ Committed: ${ordinal(n)} prompt '${clean}'\n${status}`);
  }
}

// ============================================================
// COMMIT BROWSING / TIME TRAVEL
// ============================================================

async function showCommits() {
  const commits = await listCommits();
  if (!commits.length) return console.log("\nNo commits yet.");
  console.log("\n📜 Commits (oldest → newest):");
  for (const c of commits) {
    console.log(`  ${c.hash}  ${c.subject}  (${c.when})`);
  }
  console.log('\nTo go back: type "/goto <n>" (n = prompt number) or "stay with 3rd commit".');
}

async function gotoCommit(n) {
  const commits = await listCommits();
  const target = commits.find((c) => c.promptNo === n);
  if (!target) return console.log(`\nNo commit found for prompt #${n}. Use /commits to see them.`);

  console.log(`\nTarget: ${target.hash}  ${target.subject}`);
  console.log("All later commits will be removed from this branch (a backup branch is kept).");
  if (!readlineSync.keyInYN("Continue?")) return console.log("Cancelled.");

  if (await isDirty()) await gitCommit("autosave before switching commits");
  const backup = `backup-${Date.now()}`;
  await git(["branch", backup]);
  await git(["reset", "--hard", target.hash]);
  console.log(`\n✅ Now at ${ordinal(n)} prompt commit. Previous state saved on branch "${backup}".`);
}

// ============================================================
// MAIN
// ============================================================

async function main() {
  await resolveModel();

  console.log(`
==========================================
       🚀 AI POWER CODING ASSISTANT
==========================================
Model:    ${MODEL}
Project:  ${CWD}
(the AI can only edit files inside this folder)

Commands:
  /commits         show commit history
  /goto <n>        go back to the nth prompt commit
  exit             quit

Anything else is sent to the AI.
==========================================
`);

  while (true) {
    const input = readlineSync.question("\nYou --> ").trim();
    if (!input) continue;

    try {
      if (/^(exit|quit)$/i.test(input)) {
        console.log("\nGoodbye 👋");
        break;
      }

      if (/^\/commits$/i.test(input) || /^(show|list|see)\b.*\bcommits?\b/i.test(input)) {
        await showCommits();
        continue;
      }

      const goto =
        input.match(/^\/goto\s+(\d+)/i) ||
        (/\bcommit\b/i.test(input) &&
          /\b(stay|go|back|switch|revert|restore)\b/i.test(input) &&
          input.match(/(\d+)\s*(?:st|nd|rd|th)?/));
      if (goto) {
        await gotoCommit(Number(goto[1]));
        continue;
      }

      await runAgent(input);
    } catch (err) {
      console.error(`\n❌ ${err.message}`);
    }
  }
}

main();