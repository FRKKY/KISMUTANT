# 🚀 Cloud Deployment Guide

This guide will walk you through deploying the Living Trading System to the cloud using **Railway** (recommended) or **Render**.

---

## Prerequisites

Before starting, make sure you have:

1. ✅ A GitHub account (you have this)
2. ✅ A Telegram account (you have this)
3. ⏳ KIS API credentials (get these when you open your account)
4. ⏳ Telegram bot created (we'll do this below)

---

## Step 1: Create Your Telegram Bot (5 minutes)

### 1.1 Create the Bot

1. Open Telegram and search for **@BotFather**
2. Start a chat and send: `/newbot`
3. Follow the prompts:
   - Enter a name for your bot (e.g., "My Trading System")
   - Enter a username (must end in "bot", e.g., "mytrading123_bot")
4. **Save the bot token** - it looks like: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`

### 1.2 Get Your Chat ID

1. Message your new bot (send "hello")
2. Open this URL in your browser (replace YOUR_TOKEN with your actual token):
   ```
   https://api.telegram.org/botYOUR_TOKEN/getUpdates
   ```
3. Look for `"chat":{"id":` followed by a number
4. **Save this number** - this is your Chat ID (e.g., `987654321`)

---

## Step 2: Upload Code to GitHub (10 minutes)

### 2.1 Create a New Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `living-trading-system`
3. Set to **Private** (important - your credentials might be visible)
4. Click **Create repository**

### 2.2 Upload the Code

**Option A: Using GitHub Web Interface (Easiest)**

1. On your new repository page, click **"uploading an existing file"**
2. Extract the ZIP file I gave you
3. Drag and drop ALL the files/folders into the upload area
4. Click **Commit changes**

**Option B: Using Git Command Line**

```bash
# Extract the ZIP file
unzip living-trading-system-phase1-with-interfaces.zip
cd living-trading-system

# Initialize git
git init
git add .
git commit -m "Initial commit"

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/living-trading-system.git
git branch -M main
git push -u origin main
```

---

## Step 3: Deploy to Railway (Recommended)

### 3.1 Create Railway Account

1. Go to [railway.app](https://railway.app)
2. Click **Login** → **Login with GitHub**
3. Authorize Railway to access your GitHub

### 3.2 Create New Project

1. Click **New Project**
2. Select **Deploy from GitHub repo**
3. Find and select `living-trading-system`
4. Railway will automatically detect the Dockerfile and start building

### 3.3 Configure Environment Variables

This is where you add your credentials securely (they won't be in your code).

1. Click on your deployed service
2. Go to **Variables** tab
3. Add these variables one by one:

```
# Trading Mode
TRADING_MODE=paper

# KIS Paper Trading (add these when you have your account)
KIS_PAPER_APP_KEY=your_paper_app_key
KIS_PAPER_APP_SECRET=your_paper_app_secret
KIS_PAPER_ACCOUNT_NUMBER=your_paper_account
KIS_PAPER_ACCOUNT_PRODUCT_CODE=01
KIS_HTS_ID=your_hts_id

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_from_step_1
TELEGRAM_CHAT_ID=your_chat_id_from_step_1

# Web Dashboard
PORT=8080
```

### 3.4 Generate Public URL

1. Go to **Settings** tab
2. Under **Networking**, click **Generate Domain**
3. You'll get a URL like: `living-trading-system-production.up.railway.app`
4. Your web dashboard will be at this URL

### 3.5 Verify Deployment

1. Open your Railway URL in a browser - you should see the dashboard
2. Check your Telegram - the bot should send a "System Started" message
3. Try sending `/status` to your bot

---

## Step 4: Alternative - Deploy to Render

If you prefer Render over Railway:

### 4.1 Create Render Account

1. Go to [render.com](https://render.com)
2. Click **Get Started** → **GitHub**
3. Authorize Render

### 4.2 Create New Web Service

1. Click **New** → **Web Service**
2. Connect your `living-trading-system` repository
3. Configure:
   - Name: `living-trading-system`
   - Environment: `Docker`
   - Plan: `Starter` (~$7/month)
4. Click **Create Web Service**

### 4.3 Add Environment Variables

1. Go to **Environment** tab
2. Add the same variables as listed in Step 3.3

### 4.4 Get Your URL

Your dashboard URL will be: `living-trading-system.onrender.com`

---

## Step 5: Managing Your System

### Accessing the Dashboard

- **Railway**: `https://your-project.up.railway.app`
- **Render**: `https://living-trading-system.onrender.com`

### Using Telegram Commands

| Command | What it does |
|---------|--------------|
| `/status` | Shows system overview |
| `/portfolio` | Shows portfolio details |
| `/positions` | Lists current holdings |
| `/hypotheses` | Shows active strategies |
| `/trades` | Shows recent trades |
| `/pause` | Pauses trading |
| `/resume` | Resumes trading |
| `/stop CONFIRM` | Emergency stop |
| `/help` | Lists all commands |

### Viewing Logs

**Railway:**
1. Click your service
2. Go to **Deployments** tab
3. Click **View Logs**

**Render:**
1. Click your service
2. Go to **Logs** tab

---

## Step 6: When You Get Your KIS Account

Once you have your Korea Investment & Securities account:

1. Apply for API access at [KIS Developers](https://apiportal.koreainvestment.com)
2. Get your App Key and App Secret for both:
   - 모의투자 (Paper Trading)
   - 실전투자 (Real Trading)
3. Update your environment variables in Railway/Render
4. The system will automatically reconnect

---

## Cost Estimate

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| Railway | $5/month credit | ~$5-15/month |
| Render | Limited free | ~$7/month |
| **Total** | ~$0-5/month | ~$7-15/month |

---

## Troubleshooting

### Bot not responding?

1. Check if the service is running (Railway/Render dashboard)
2. Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are correct
3. Check the logs for errors

### Dashboard not loading?

1. Make sure port 8080 is exposed
2. Check if a public URL was generated
3. Look at deployment logs for errors

### KIS API errors?

1. Verify all KIS credentials are entered correctly
2. Check if you're using paper or real credentials for the current mode
3. API might be down during KRX maintenance hours

### Need to restart?

**Railway:** Go to Deployments → Click "Redeploy"
**Render:** Go to your service → Click "Manual Deploy"

---

## Security Notes

⚠️ **Important Security Practices:**

1. **Never commit credentials to GitHub** - always use environment variables
2. **Keep your repository private** - even without credentials, your strategy code is valuable
3. **Use paper trading first** - always test with fake money before real trading
4. **Monitor your bot** - set up alerts for unusual activity

---

## Next Steps

Once deployed and running:

1. ✅ Verify web dashboard works
2. ✅ Verify Telegram bot responds
3. ⏳ Wait for KIS account approval
4. ⏳ Add KIS credentials to environment variables
5. ⏳ Test with paper trading for 2-3 months
6. ⏳ Graduate to live trading when ready

---

## Questions?

If you run into issues, check:
1. Railway/Render deployment logs
2. The troubleshooting section above
3. Ask me for help!
