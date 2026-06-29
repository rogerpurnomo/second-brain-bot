"""
Roger's Second Brain — an autonomous idea-development AGENT on Telegram.

You just talk to it in natural language. Gemini decides, on its own, when to
call tools — develop an idea, save it, search the vault, connect it to old
notes, update or delete a note. Notes are written as markdown to an Obsidian
vault repo on GitHub, which the Obsidian Git plugin pulls within ~5 minutes.

Architecture:
- Stateful bot, stateless model: per-user conversation history is kept in
  memory and replayed each turn.
- Agentic loop: model -> (optional) tool calls -> tool results -> model ...
  until it returns a final message. The MODEL chooses the tools.
- Gemini is called via its OpenAI-compatible chat-completions endpoint.
"""

import base64
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

# Local convenience: load a .env file if python-dotenv is installed.
# In production (Render) real env vars are injected, so this is a no-op.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# --------------------------------------------------------------------------- #
# Config (all from environment — see .env.example)
# --------------------------------------------------------------------------- #
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPO = os.environ["GITHUB_REPO"]          # "username/vault-repo"
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
ALLOWED_USER_ID = int(os.environ["ALLOWED_USER_ID"])

# Gemini via its OpenAI-compatible endpoint. Override the model if it's retired.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GITHUB_API = "https://api.github.com"

MAX_TOOL_ITERATIONS = 6     # safety cap on the agent loop per user turn
MAX_HISTORY = 24            # messages kept per user before trimming

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("second-brain")

# Per-user state: {"step": str, "history": list[dict], "idea_files": ..., "edit_target": ...}
user_state: dict[int, dict] = {}


# --------------------------------------------------------------------------- #
# Agent persona + tool definitions
# --------------------------------------------------------------------------- #
AGENT_SYSTEM = """You are Roger's Second Brain — an autonomous idea-development agent living in his Telegram.

About Roger: Data Analytics student at Asia Pacific University (APU), Malaysia. ADHD + overthinker — brilliant ideas that vanish fast. In APU AI Club (AIC), building ClearLedge (cross-border payments), runs Martabak Bangka 66 (food biz). Into Web3, investing, Indonesian stocks (IDX), AI/ML, data pipelines.

Your job: help him capture and develop ideas, and manage his vault — and YOU decide when to act using your tools. You don't wait for buttons or explicit commands.

Your tools:
- save_idea — keep a developed idea permanently in the Ideas folder
- quick_capture — dump a raw thought to the Inbox with no development
- search_vault / read_idea — find and read existing notes (use these to connect new ideas to old ones)
- list_recent_ideas — see recent notes
- update_idea — append to an existing note
- delete_idea — remove a note (ALWAYS confirm with Roger in your reply BEFORE calling it)

How to behave:
- HIGH ENERGY and hype — match his excitement. ADHD-friendly: punchy, scannable, short. Bullets over paragraphs.
- Be Socratic: usually end with the ONE most important question.
- Be practical: next steps must be doable THIS WEEK.
- Act proactively but sensibly. Develop an idea when he shares one; when it's solid or he signals he's done, call save_idea without making him ask twice. When something feels related to past work, search_vault and connect it.
- Don't save on every tiny message — save when there's something worth keeping or he asks.
- Deletes: confirm first; only call delete_idea after he says yes.
- After you save/update/delete, tell him what you did and the note title.
- Use his context (AIC, ClearLedge, Martabak, Web3, IDX) when relevant."""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "save_idea",
            "description": (
                "Save a developed idea as a permanent note in the Ideas folder. "
                "Use when Roger wants to keep an idea, or after you've developed it "
                "together and it's worth keeping."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "A punchy 4-6 word title"},
                    "raw_idea": {"type": "string", "description": "What Roger originally said, unfiltered"},
                    "development": {
                        "type": "string",
                        "description": "The developed thinking as markdown: key points, structure, insights, next steps",
                    },
                },
                "required": ["title", "raw_idea", "development"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "quick_capture",
            "description": (
                "Instantly save a raw thought to the Inbox without developing it. "
                "Use when Roger says 'just save this', has no time, or wants a quick capture."
            ),
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_vault",
            "description": (
                "Search saved notes by keyword to find related or specific ideas. Use to "
                "connect a new idea to existing ones, or to locate a note before reading/"
                "updating/deleting it."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_recent_ideas",
            "description": "List the filenames of the most recent saved ideas.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_idea",
            "description": "Read the full content of a saved idea by its filename (from search or list).",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_idea",
            "description": "Append a dated update to an existing idea note, by filename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["filename", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_idea",
            "description": (
                "Permanently delete an idea note by filename. ALWAYS ask Roger to confirm "
                "in your reply BEFORE calling this — never delete without an explicit yes."
            ),
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Auth guard
# --------------------------------------------------------------------------- #
def authorized(update: Update) -> bool:
    user = update.effective_user
    return user is not None and user.id == ALLOWED_USER_ID


# --------------------------------------------------------------------------- #
# GitHub vault integration (REST API — no git CLI on the server)
# --------------------------------------------------------------------------- #
def _gh_headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def gh_get_file(path: str) -> tuple[str | None, str | None]:
    """Read a file from the vault repo. Returns (text_content, sha) or (None, None)."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}",
            headers=_gh_headers(),
            params={"ref": GITHUB_BRANCH},
        )
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


async def gh_put_file(path: str, content: str, message: str, sha: str | None = None) -> None:
    """Create or update a file in the vault repo."""
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.put(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}",
            headers=_gh_headers(),
            json=payload,
        )
    resp.raise_for_status()


async def gh_delete_file(path: str, sha: str, message: str) -> None:
    """Delete a file from the vault repo (needs its blob sha)."""
    payload = {"message": message, "sha": sha, "branch": GITHUB_BRANCH}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(
            "DELETE",
            f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{path}",
            headers=_gh_headers(),
            json=payload,
        )
    resp.raise_for_status()


async def gh_list_folder(folder: str) -> list[dict]:
    """List .md files in a vault folder, newest-first by name."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{folder}",
            headers=_gh_headers(),
            params={"ref": GITHUB_BRANCH},
        )
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    files = [f for f in resp.json() if f["name"].endswith(".md")]
    files.sort(key=lambda f: f["name"], reverse=True)
    return files


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:60] or "idea"


async def save_idea_to_vault(title: str, raw_idea: str, development: str) -> str:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    note = f"""# {title}

Date: {date}
Status: #seed
Tags:

## The Raw Idea
{raw_idea}

## AI Session
{development}

## Next Steps
- [ ]

## Related Ideas
"""
    path = f"Ideas/{date}-{slugify(title)}.md"
    await gh_put_file(path, note, message=f"Add idea: {title}")
    return path


async def save_to_inbox(raw_text: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    note = f"""# Quick Capture

Date: {ts}

{raw_text}
"""
    path = f"Inbox/inbox-{ts}.md"
    await gh_put_file(path, note, message="Quick capture to inbox")
    return path


# --------------------------------------------------------------------------- #
# Agent tools (the functions the model can call)
# --------------------------------------------------------------------------- #
def _norm_filename(name: str) -> str:
    name = name.strip().rsplit("/", 1)[-1]
    return name if name.endswith(".md") else f"{name}.md"


async def tool_save_idea(title: str, raw_idea: str, development: str) -> str:
    path = await save_idea_to_vault(title, raw_idea, development)
    return f"Saved as '{title}' at {path}"


async def tool_quick_capture(text: str) -> str:
    path = await save_to_inbox(text)
    return f"Quick-captured to {path}"


async def tool_list_recent_ideas() -> str:
    files = await gh_list_folder("Ideas")
    if not files:
        return "No ideas saved yet."
    return "\n".join(f["name"] for f in files[:10])


async def tool_search_vault(query: str) -> str:
    files = await gh_list_folder("Ideas")
    q = query.lower()
    hits = []
    for f in files[:20]:
        content, _ = await gh_get_file(f["path"])
        if content and (q in content.lower() or q in f["name"].lower()):
            line = next((ln for ln in content.splitlines() if q in ln.lower() and ln.strip()), "")
            hits.append(f"{f['name']}: {line.strip()[:120]}")
        if len(hits) >= 6:
            break
    return "\n".join(hits) if hits else f"No notes matching '{query}'."


async def tool_read_idea(filename: str) -> str:
    content, _ = await gh_get_file(f"Ideas/{_norm_filename(filename)}")
    return content or f"No note named {filename}."


async def tool_update_idea(filename: str, text: str) -> str:
    path = f"Ideas/{_norm_filename(filename)}"
    content, sha = await gh_get_file(path)
    if content is None:
        return f"No note named {filename}."
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_content = f"{content.rstrip()}\n\n## Update {date}\n{text}\n"
    await gh_put_file(path, new_content, "Update idea (agent)", sha)
    return f"Appended an update to {path}"


async def tool_delete_idea(filename: str) -> str:
    path = f"Ideas/{_norm_filename(filename)}"
    _, sha = await gh_get_file(path)
    if sha is None:
        return f"No note named {filename}."
    await gh_delete_file(path, sha, "Delete idea (agent)")
    return f"Deleted {path}"


TOOL_FUNCS = {
    "save_idea": tool_save_idea,
    "quick_capture": tool_quick_capture,
    "search_vault": tool_search_vault,
    "list_recent_ideas": tool_list_recent_ideas,
    "read_idea": tool_read_idea,
    "update_idea": tool_update_idea,
    "delete_idea": tool_delete_idea,
}


async def dispatch_tool(name: str, args: dict) -> str:
    func = TOOL_FUNCS.get(name)
    if func is None:
        return f"Unknown tool: {name}"
    try:
        return await func(**args)
    except TypeError as exc:
        return f"Bad arguments for {name}: {exc}"
    except Exception as exc:  # surface the error to the model so it can recover
        logger.exception("tool %s failed", name)
        return f"Error running {name}: {exc}"


# --------------------------------------------------------------------------- #
# Gemini (OpenAI-compatible chat completions) + agent loop
# --------------------------------------------------------------------------- #
async def _gemini_post(payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(GEMINI_URL, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


async def run_agent(history: list[dict]) -> str:
    """Run the model->tools->model loop until a final text answer. Mutates history."""
    for _ in range(MAX_TOOL_ITERATIONS):
        data = await _gemini_post(
            {
                "model": GEMINI_MODEL,
                "max_tokens": 1500,
                "messages": [{"role": "system", "content": AGENT_SYSTEM}, *history],
                "tools": TOOL_SCHEMAS,
                "tool_choice": "auto",
                "reasoning_effort": "none",
            }
        )
        msg = data["choices"][0]["message"]

        entry = {"role": "assistant", "content": msg.get("content")}
        if msg.get("tool_calls"):
            entry["tool_calls"] = msg["tool_calls"]
        history.append(entry)

        if not msg.get("tool_calls"):
            return msg.get("content") or "(no response)"

        for tc in msg["tool_calls"]:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await dispatch_tool(tc["function"]["name"], args)
            history.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": str(result)}
            )

    return "I went in circles there — mind rephrasing?"


def trim_history(history: list[dict]) -> list[dict]:
    """Cap history length, cutting at a clean user-message boundary."""
    if len(history) <= MAX_HISTORY:
        return history
    history = history[-MAX_HISTORY:]
    while history and history[0].get("role") != "user":
        history.pop(0)
    return history


# --------------------------------------------------------------------------- #
# Command handlers
# --------------------------------------------------------------------------- #
WELCOME = (
    "🧠 *Second Brain online.*\n\n"
    "Just talk to me — I'll develop your ideas, connect them to old ones, and "
    "save/search/update/delete notes on my own as we chat. No buttons needed.\n\n"
    "Try: _“I want to build a dividend tracker for IDX stocks”_ — or _“what have I "
    "saved about ClearLedge?”_\n\n"
    "*Shortcuts*\n"
    "/inbox — quick-save a raw thought\n"
    "/list — show recent saved ideas\n"
    "/edit — add an update to an idea\n"
    "/delete — remove an idea\n"
    "/reset — start a fresh conversation\n"
    "/help — this message"
)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    user_state[update.effective_user.id] = {"step": "chatting", "history": []}
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    user_state[update.effective_user.id] = {"step": "chatting", "history": []}
    await update.message.reply_text("🧹 Fresh start. What's on your mind?")


async def cmd_inbox(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    text = update.message.text.partition(" ")[2].strip()
    if not text:
        user_state.setdefault(update.effective_user.id, {})["step"] = "awaiting_inbox"
        await update.message.reply_text("📥 Send the text to quick-save.")
        return
    path = await save_to_inbox(text)
    await update.message.reply_text(f"📥 Saved to `{path}`", parse_mode=ParseMode.MARKDOWN)


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    files = await gh_list_folder("Ideas")
    if not files:
        await update.message.reply_text("No saved ideas yet. Send me one!")
        return
    lines = ["🗂 *Recent ideas*"]
    for f in files[:10]:
        lines.append(f"• {f['name'].removesuffix('.md')}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


def build_idea_list_keyboard(files: list[dict], prefix: str) -> InlineKeyboardMarkup:
    """One button per idea; callback data is f'{prefix}:{index}' into the list."""
    rows = [
        [InlineKeyboardButton(f["name"].removesuffix(".md")[:45], callback_data=f"{prefix}:{i}")]
        for i, f in enumerate(files)
    ]
    return InlineKeyboardMarkup(rows)


async def cmd_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    files = (await gh_list_folder("Ideas"))[:10]
    if not files:
        await update.message.reply_text("No ideas to delete yet.")
        return
    user_state.setdefault(update.effective_user.id, {})["idea_files"] = files
    await update.message.reply_text(
        "🗑 Which idea to delete?", reply_markup=build_idea_list_keyboard(files, "del")
    )


async def cmd_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    files = (await gh_list_folder("Ideas"))[:10]
    if not files:
        await update.message.reply_text("No ideas to edit yet.")
        return
    user_state.setdefault(update.effective_user.id, {})["idea_files"] = files
    await update.message.reply_text(
        "✏️ Which idea do you want to add to?",
        reply_markup=build_idea_list_keyboard(files, "ed"),
    )


def _picked_file(uid: int, data: str) -> dict | None:
    try:
        idx = int(data.split(":", 1)[1])
    except (ValueError, IndexError):
        return None
    files = user_state.get(uid, {}).get("idea_files") or []
    return files[idx] if 0 <= idx < len(files) else None


async def on_delete_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.from_user.id != ALLOWED_USER_ID:
        return
    f = _picked_file(query.from_user.id, query.data)
    if not f:
        await query.edit_message_text("That list expired — run /delete again.")
        return
    name = f["name"].removesuffix(".md")
    idx = query.data.split(":", 1)[1]
    keyboard = InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("✅ Yes, delete", callback_data=f"delyes:{idx}"),
            InlineKeyboardButton("❌ Cancel", callback_data="delcancel"),
        ]]
    )
    await query.edit_message_text(
        f"Delete *{name}*? This can't be undone.",
        reply_markup=keyboard,
        parse_mode=ParseMode.MARKDOWN,
    )


async def on_delete_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.from_user.id != ALLOWED_USER_ID:
        return
    f = _picked_file(query.from_user.id, query.data)
    if not f:
        await query.edit_message_text("That list expired — run /delete again.")
        return
    name = f["name"].removesuffix(".md")
    await gh_delete_file(f["path"], f["sha"], f"Delete idea: {name}")
    await query.edit_message_text(f"🗑 Deleted *{name}*", parse_mode=ParseMode.MARKDOWN)


async def on_delete_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Cancelled — nothing deleted.")


async def on_edit_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.from_user.id != ALLOWED_USER_ID:
        return
    uid = query.from_user.id
    f = _picked_file(uid, query.data)
    if not f:
        await query.edit_message_text("That list expired — run /edit again.")
        return
    name = f["name"].removesuffix(".md")
    state = user_state.setdefault(uid, {"step": "chatting", "history": []})
    state["step"] = "awaiting_edit_text"
    state["edit_target"] = {"path": f["path"], "name": name}
    await query.edit_message_text(
        f"✏️ Editing *{name}*\nSend the text to add — I'll append it as a dated update.",
        parse_mode=ParseMode.MARKDOWN,
    )


# --------------------------------------------------------------------------- #
# Message handler — the agent
# --------------------------------------------------------------------------- #
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    uid = update.effective_user.id
    text = update.message.text.strip()
    state = user_state.setdefault(uid, {"step": "chatting", "history": []})
    chat_id = update.effective_chat.id

    # /edit follow-up: append the text to the chosen note
    if state.get("step") == "awaiting_edit_text":
        target = state.get("edit_target")
        content, sha = (await gh_get_file(target["path"])) if target else (None, None)
        state["step"] = "chatting"
        if content is None:
            await update.message.reply_text("That note no longer exists.")
            return
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        new_content = f"{content.rstrip()}\n\n## Update {date}\n{text}\n"
        await gh_put_file(target["path"], new_content, f"Update idea: {target['name']}", sha)
        await update.message.reply_text(
            f"✏️ Added your update to *{target['name']}*", parse_mode=ParseMode.MARKDOWN
        )
        return

    # /inbox follow-up: quick-save
    if state.get("step") == "awaiting_inbox":
        state["step"] = "chatting"
        path = await save_to_inbox(text)
        await update.message.reply_text(f"📥 Saved to `{path}`", parse_mode=ParseMode.MARKDOWN)
        return

    # Default: hand the message to the agent
    history = state.setdefault("history", [])
    history.append({"role": "user", "content": text})
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        reply = await run_agent(history)
    except Exception:
        logger.exception("agent failed")
        await update.message.reply_text(
            "⚠️ Something glitched on my end — try again in a sec."
        )
        return
    state["history"] = trim_history(history)
    # Plain text (no Markdown) — the model's free-form output can break TG's parser.
    await context.bot.send_message(chat_id=chat_id, text=reply or "…")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled error", exc_info=context.error)


# --------------------------------------------------------------------------- #
# Health-check server (Render/Koyeb require an open port)
# --------------------------------------------------------------------------- #
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def start_health_server() -> None:
    port = int(os.environ.get("PORT", "8000"))
    try:
        server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    except OSError as exc:
        logger.warning("Health server not started on :%s (%s)", port, exc)
        return
    logger.info("Health server listening on :%s", port)
    threading.Thread(target=server.serve_forever, daemon=True).start()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    start_health_server()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("inbox", cmd_inbox))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("delete", cmd_delete))
    app.add_handler(CommandHandler("edit", cmd_edit))
    app.add_handler(CallbackQueryHandler(on_delete_confirm, pattern=r"^delyes:"))
    app.add_handler(CallbackQueryHandler(on_delete_cancel, pattern=r"^delcancel$"))
    app.add_handler(CallbackQueryHandler(on_delete_pick, pattern=r"^del:"))
    app.add_handler(CallbackQueryHandler(on_edit_pick, pattern=r"^ed:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_error_handler(on_error)

    logger.info("Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
