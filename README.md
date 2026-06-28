# Roger's Second Brain — Telegram Bot

Capture ideas from your phone, develop them with Claude, and save structured
`.md` notes straight into your Obsidian vault (via a GitHub repo the Obsidian
Git plugin auto-pulls).

```
idea → Telegram → pick a mode → Claude develops it → 💾 → GitHub → Obsidian
```

## Commands

| Command          | Function                          |
| ---------------- | --------------------------------- |
| `/start`, `/help`| Welcome + command list            |
| `/idea`          | Start developing an idea          |
| `/inbox [text]`  | Quick-save without developing     |
| `/list`          | Show 10 most recent saved ideas   |

Drop a raw idea, pick a development mode (🧠 Brain Dump, 🔗 Connect, 🔥 Pressure
Test, 🚀 Next Steps, or 📥 Quick Save), chat with Claude, then tap 💾 Save,
🔄 Keep going, or ✅ Done.

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
