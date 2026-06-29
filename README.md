# Roger's Second Brain — Telegram Bot

An autonomous **AI agent** on Telegram. You talk to it in plain language and it
**decides on its own** when to develop an idea, save it, search your vault,
connect new ideas to old ones, or update/delete notes — using Gemini
function-calling. Notes are written as structured `.md` files to your Obsidian
vault (a GitHub repo the Obsidian Git plugin auto-pulls).

```
you talk → Gemini agent decides + calls tools → GitHub vault → Obsidian
```

The agent's tools: `save_idea`, `quick_capture`, `search_vault`, `read_idea`,
`list_recent_ideas`, `update_idea`, `delete_idea`. The model picks which to call.

## Commands

| Command          | Function                          |
| ---------------- | --------------------------------- |
| `/start`, `/help`| Welcome + command list            |
| `/idea`          | Start developing an idea          |
| `/inbox [text]`  | Quick-save without developing     |
| `/list`          | Show 10 most recent saved ideas   |
| `/edit`          | Append a dated update to an idea  |
| `/delete`        | Remove an idea (with confirm)     |

Mostly you just chat — no buttons. Say things like *"develop this idea and save
it"*, *"what have I saved about ClearLedge?"*, or *"delete the parking one"* and
the agent figures out which tools to call. The slash commands above are optional
shortcuts (`/delete` and `/edit` still offer button pickers).

## Tech stack (100% free tier)

| Layer        | Tool                          |
| ------------ | ----------------------------- |
| Bot          | Telegram (python-telegram-bot)|
| Brain        | Gemini API (`gemini-2.5-flash`, free tier) |
| Vault sync   | GitHub private repo (REST API)|
| Hosting      | Railway.app                   |

## Environment variables

See `.env.example`. All are required except `GITHUB_BRANCH` (defaults `main`)
and `GEMINI_MODEL` (defaults `gemini-2.5-flash`).

| Var                | Where to get it                                      |
| ------------------ | ---------------------------------------------------- |
| `TELEGRAM_TOKEN`   | @BotFather → /newbot                                 |
| `GEMINI_API_KEY`   | aistudio.google.com/apikey (free)                    |
| `GITHUB_TOKEN`     | GitHub → fine-grained PAT, Contents read/write       |
| `GITHUB_REPO`      | `yourusername/vault-repo-name`                       |
| `GITHUB_BRANCH`    | usually `main`                                       |
| `ALLOWED_USER_ID`  | @userinfobot → your numeric ID                       |

## Vault repo layout

```
Inbox/      ← quick captures
Ideas/      ← developed ideas saved from the bot
Projects/   ← (manual) active work
Resources/  ← (manual) reference material
```

## Run locally

```bash
cd second-brain-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install python-dotenv          # optional, for loading .env locally
cp .env.example .env               # then fill it in
python bot.py
```

You should see `Bot is running...`. Message your bot on Telegram to test.

## Deploy to Railway

1. Push this folder to its own GitHub repo (separate from the vault).
2. Railway → New Project → Deploy from GitHub repo → pick it.
3. Variables tab → add every var from `.env.example`.
4. Deploy. Check logs for `Bot is running...`.

Railway reads `railway.toml` (`python bot.py`) and installs `requirements.txt`
automatically via Nixpacks. Only one instance should run at a time (polling).

## Setup order (first time)

1. Create a private GitHub repo for the Obsidian vault with the 4 folders.
2. In Obsidian, install the Git plugin and set auto-pull every 5 min.
3. Create the Telegram bot via @BotFather; get your user ID via @userinfobot.
4. Create a GitHub fine-grained PAT (Contents read/write on the vault repo).
5. Push this bot to its own repo, connect to Railway, set vars, deploy.
