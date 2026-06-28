"""
Roger's Second Brain — Telegram capture + development bot.

Flow:
  raw idea -> pick a development mode (inline buttons) -> an LLM develops it
  with you -> tap 💾 to save a structured .md note to your Obsidian vault repo
  on GitHub, which the Obsidian Git plugin pulls within ~5 minutes.

Stateless LLM API, stateful bot: per-user conversation history is held in
memory and replayed to the model each turn.
"""

import base64
import logging
import os
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

# Local convenience: load a .env file if python-dotenv is installed.
# In production (Railway) real env vars are injected, so this is a no-op.
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

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("second-brain")

# Per-user state machine. Steps: awaiting_idea -> awaiting_mode -> in_conversation
user_state: dict[int, dict] = {}

# --------------------------------------------------------------------------- #
# Agent personality + modes
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are Roger's Second Brain — his idea-development partner.

About Roger: Data Analytics student at Asia Pacific University (APU), Malaysia.
ADHD + overthinker — great ideas that pass by fast. Involved in APU AI Club (AIC),
building ClearLedge (cross-border payment tool), runs Martabak Bangka 66 (food biz).
Into Web3, investing, Indonesian stocks (IDX), AI/ML, data pipelines.

Your personality:
- HIGH ENERGY and hype — match Roger's excitement.
- Socratic — always end by asking the ONE most important question.
- Practical — next steps must be doable THIS WEEK, not someday.
- ADHD-friendly — punchy, scannable, short. No walls of text. Use bullets.
- You know his context (AIC, ClearLedge, Martabak, Web3, IDX) — use it.

Keep replies tight. Telegram-friendly formatting."""

MODES = {
    "brain_dump": {
        "label": "🧠 Brain Dump → Structure",
        "instruction": (
            "Roger dropped a messy idea. Help him make sense of it. Pull out the "
            "core insight, structure the moving parts, and reflect it back cleanly. "
            "Then ask the ONE question that unlocks it most."
        ),
    },
    "connect": {
        "label": "🔗 Connect to Existing Ideas",
        "instruction": (
            "This idea feels related to things already in Roger's vault. Using the "
            "recent ideas provided as context, surface the strongest connections and "
            "what they imply together. Then ask the ONE most important question."
        ),
    },
    "pressure_test": {
        "label": "🔥 Pressure Test",
        "instruction": (
            "Roger is excited about this. Give an honest, high-energy stress-test: "
            "the strongest version, the real risks, and the fastest way to validate "
            "it cheaply. Be direct but supportive. End with the ONE question."
        ),
    },
    "next_steps": {
        "label": "🚀 Next Steps",
        "instruction": (
            "This idea is developed enough to act on. Give 3-5 concrete next steps "
            "Roger can actually do THIS WEEK. Be specific. End with the ONE first "
            "action to take today."
        ),
    },
}

# Inline keyboard shown after a raw idea (📥 quick-save is handled separately)
MODE_KEYBOARD = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton(MODES["brain_dump"]["label"], callback_data="mode:brain_dump")],
        [InlineKeyboardButton(MODES["connect"]["label"], callback_data="mode:connect")],
        [InlineKeyboardButton(MODES["pressure_test"]["label"], callback_data="mode:pressure_test")],
        [InlineKeyboardButton(MODES["next_steps"]["label"], callback_data="mode:next_steps")],
        [InlineKeyboardButton("📥 Quick Save to Inbox", callback_data="mode:inbox")],
    ]
)

# Inline keyboard shown after Claude replies
ACTION_KEYBOARD = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("💾 Save", callback_data="act:save"),
            InlineKeyboardButton("🔄 Keep going", callback_data="act:keep"),
            InlineKeyboardButton("✅ Done", callback_data="act:done"),
        ]
    ]
)


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


async def load_recent_ideas(max_ideas: int = 8) -> str:
    """Load the last N Ideas as short context snippets for Claude."""
    files = await gh_list_folder("Ideas")
    snippets = []
    for f in files[:max_ideas]:
        content, _ = await gh_get_file(f["path"])
        if content:
            snippets.append(content[:600])
    if not snippets:
        return "(No saved ideas yet.)"
    return "\n\n---\n\n".join(snippets)


# --------------------------------------------------------------------------- #
# LLM brain (Gemini — OpenAI-compatible chat completions, direct HTTP)
# --------------------------------------------------------------------------- #
async def call_llm(messages: list[dict], system: str, max_tokens: int = 1024) -> str:
    payload = {
        "model": GEMINI_MODEL,
        "max_tokens": max_tokens,
        # OpenAI-style: system prompt is the first message in the list
        "messages": [{"role": "system", "content": system}, *messages],
        # Gemini 2.5 models "think" by default, which eats the token budget
        # before any visible answer. Turn it off so output goes to the reply.
        "reasoning_effort": "none",
    }
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(GEMINI_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def generate_title(raw_idea: str, claude_output: str) -> str:
    """Ask Claude for a 4-6 word title for the note filename."""
    prompt = (
        "Give a 4-6 word title for this idea. Plain text only, no quotes, no "
        f"punctuation at the end.\n\nIdea: {raw_idea}\n\nDevelopment:\n{claude_output[:1500]}"
    )
    title = await call_llm(
        [{"role": "user", "content": prompt}],
        system="You write short, punchy note titles.",
        max_tokens=30,
    )
    return title.strip().splitlines()[0].strip().strip('"') or "Untitled Idea"


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:60] or "idea"


async def save_idea_to_vault(title: str, raw_idea: str, claude_output: str) -> str:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    note = f"""# {title}

Date: {date}
Status: #seed
Tags:

## The Raw Idea
{raw_idea}

## AI Session
{claude_output}

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
# Command handlers
# --------------------------------------------------------------------------- #
WELCOME = (
    "🧠 *Second Brain online.*\n\n"
    "Drop an idea and I'll help you develop it before it slips away.\n\n"
    "*Commands*\n"
    "/idea — develop an idea with me\n"
    "/inbox — quick-save without developing\n"
    "/list — show 10 most recent saved ideas\n"
    "/edit — add an update to an existing idea\n"
    "/delete — remove an idea\n"
    "/help — this message\n\n"
    "Or just send me any message to start."
)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    user_state[update.effective_user.id] = {"step": "awaiting_idea"}
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN)


async def cmd_idea(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    user_state[update.effective_user.id] = {"step": "awaiting_idea"}
    await update.message.reply_text("💡 Hit me — what's the idea?")


async def cmd_inbox(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    text = update.message.text.partition(" ")[2].strip()
    if not text:
        user_state[update.effective_user.id] = {"step": "awaiting_inbox"}
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
        name = f["name"].removesuffix(".md")
        lines.append(f"• {name}")
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
    user_state[update.effective_user.id] = {"step": "awaiting_idea", "idea_files": files}
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
    user_state[update.effective_user.id] = {"step": "awaiting_idea", "idea_files": files}
    await update.message.reply_text(
        "✏️ Which idea do you want to add to?",
        reply_markup=build_idea_list_keyboard(files, "ed"),
    )


def _picked_file(uid: int, data: str) -> dict | None:
    """Resolve a 'prefix:index' callback to the stored file dict, or None if stale."""
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
    user_state[uid] = {
        "step": "awaiting_edit_text",
        "edit_target": {"path": f["path"], "name": name},
    }
    await query.edit_message_text(
        f"✏️ Editing *{name}*\nSend the text to add — I'll append it as a dated update.",
        parse_mode=ParseMode.MARKDOWN,
    )


# --------------------------------------------------------------------------- #
# Message + callback handlers
# --------------------------------------------------------------------------- #
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    uid = update.effective_user.id
    text = update.message.text.strip()
    state = user_state.setdefault(uid, {"step": "awaiting_idea"})

    # Append an update to an existing idea (from /edit)
    if state.get("step") == "awaiting_edit_text":
        target = state.get("edit_target")
        content, sha = (await gh_get_file(target["path"])) if target else (None, None)
        if content is None:
            user_state[uid] = {"step": "awaiting_idea"}
            await update.message.reply_text("That note no longer exists.")
            return
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        new_content = f"{content.rstrip()}\n\n## Update {date}\n{text}\n"
        await gh_put_file(target["path"], new_content, f"Update idea: {target['name']}", sha)
        user_state[uid] = {"step": "awaiting_idea"}
        await update.message.reply_text(
            f"✏️ Added your update to *{target['name']}*", parse_mode=ParseMode.MARKDOWN
        )
        return

    # Quick-save flow triggered by /inbox with no argument
    if state.get("step") == "awaiting_inbox":
        path = await save_to_inbox(text)
        user_state[uid] = {"step": "awaiting_idea"}
        await update.message.reply_text(f"📥 Saved to `{path}`", parse_mode=ParseMode.MARKDOWN)
        return

    # Continue an in-progress conversation
    if state.get("step") == "in_conversation":
        await develop_idea(update, context, uid, user_text=text)
        return

    # Otherwise this is a fresh raw idea — ask for a mode
    user_state[uid] = {"step": "awaiting_mode", "raw_idea": text, "history": []}
    await update.message.reply_text(
        "Got it. How do you want to work this? 👇", reply_markup=MODE_KEYBOARD
    )


async def on_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.from_user.id != ALLOWED_USER_ID:
        return
    uid = query.from_user.id
    mode = query.data.split(":", 1)[1]
    state = user_state.get(uid)
    if not state or "raw_idea" not in state:
        await query.edit_message_text("That idea expired — send it again.")
        return

    if mode == "inbox":
        path = await save_to_inbox(state["raw_idea"])
        user_state[uid] = {"step": "awaiting_idea"}
        await query.edit_message_text(f"📥 Saved to `{path}`", parse_mode=ParseMode.MARKDOWN)
        return

    state["mode"] = mode
    state["step"] = "in_conversation"
    await query.edit_message_text(f"{MODES[mode]['label']} — let's go.")
    await develop_idea(update, context, uid, user_text=None)


async def develop_idea(
    update: Update, context: ContextTypes.DEFAULT_TYPE, uid: int, user_text: str | None
) -> None:
    state = user_state[uid]
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    history: list[dict] = state["history"]

    # First turn: seed with the raw idea + the chosen mode's instruction
    if not history:
        mode = state["mode"]
        context_block = ""
        if mode == "connect":
            recent = await load_recent_ideas()
            context_block = f"\n\nRecent ideas from Roger's vault:\n{recent}"
        seed = (
            f"{MODES[mode]['instruction']}\n\nRoger's raw idea:\n{state['raw_idea']}"
            f"{context_block}"
        )
        history.append({"role": "user", "content": seed})
    else:
        history.append({"role": "user", "content": user_text})

    reply = await call_llm(history, system=SYSTEM_PROMPT)
    history.append({"role": "assistant", "content": reply})
    state["last_response"] = reply

    await context.bot.send_message(
        chat_id=chat_id,
        text=reply,
        reply_markup=ACTION_KEYBOARD,
    )


async def on_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.from_user.id != ALLOWED_USER_ID:
        return
    uid = query.from_user.id
    action = query.data.split(":", 1)[1]
    state = user_state.get(uid)
    if not state:
        await query.edit_message_reply_markup(reply_markup=None)
        return

    if action == "save":
        await query.edit_message_reply_markup(reply_markup=None)
        await context.bot.send_chat_action(
            chat_id=query.message.chat_id, action=ChatAction.TYPING
        )
        raw = state.get("raw_idea", "")
        output = state.get("last_response", "")
        title = await generate_title(raw, output)
        path = await save_idea_to_vault(title, raw, output)
        user_state[uid] = {"step": "awaiting_idea"}
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"💾 Saved *{title}*\n`{path}`\nObsidian will pull it within ~5 min.",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif action == "keep":
        await query.edit_message_reply_markup(reply_markup=None)
        await context.bot.send_message(
            chat_id=query.message.chat_id, text="🔄 Keep going — what's on your mind?"
        )

    elif action == "done":
        user_state[uid] = {"step": "awaiting_idea"}
        await query.edit_message_reply_markup(reply_markup=None)
        await context.bot.send_message(
            chat_id=query.message.chat_id, text="✅ Done. Drop the next one whenever."
        )


# --------------------------------------------------------------------------- #
# Health-check server (for platforms like Koyeb that require an open port)
# --------------------------------------------------------------------------- #
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):  # silence per-request logging
        pass


def start_health_server() -> None:
    """Serve 200 on the platform-provided $PORT so health checks pass.

    Runs in a daemon thread alongside the polling bot. No-op if it can't bind
    (e.g. the port is already in use locally) — the bot still runs.
    """
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
    app.add_handler(CommandHandler("idea", cmd_idea))
    app.add_handler(CommandHandler("inbox", cmd_inbox))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("delete", cmd_delete))
    app.add_handler(CommandHandler("edit", cmd_edit))
    app.add_handler(CallbackQueryHandler(on_mode, pattern=r"^mode:"))
    app.add_handler(CallbackQueryHandler(on_action, pattern=r"^act:"))
    app.add_handler(CallbackQueryHandler(on_delete_confirm, pattern=r"^delyes:"))
    app.add_handler(CallbackQueryHandler(on_delete_cancel, pattern=r"^delcancel$"))
    app.add_handler(CallbackQueryHandler(on_delete_pick, pattern=r"^del:"))
    app.add_handler(CallbackQueryHandler(on_edit_pick, pattern=r"^ed:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    logger.info("Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
