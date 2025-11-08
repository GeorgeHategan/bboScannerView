# Render Deployment Guide

## Prerequisites
- GitHub repository: `GeorgeHategan/candlestick-screener`
- Render account connected to GitHub

## Deployment Steps

### 1. Create New Web Service on Render
- Go to https://render.com/dashboard
- Click **New +** → **Web Service**
- Connect to `GeorgeHategan/candlestick-screener` repository
- Render will auto-detect `render.yaml` configuration

### 2. Configure Environment Variables
Set these in Render Dashboard → Environment:

| Variable | Value | Required |
|----------|-------|----------|
| `DUCKDB_PATH` | MotherDuck connection string or path | ✅ **Critical** |
| `ALPHA_VANTAGE_API_KEY` | Your Alpha Vantage API key | ✅ **Critical** |
| `DEBUG` | `False` (for production) | Optional |

**IMPORTANT:** Since Render uses ephemeral filesystem, you MUST use MotherDuck (cloud DuckDB):
```
DUCKDB_PATH=md:?motherduck_token=YOUR_TOKEN
```
Or provide a persistent database URL.

### 3. Deploy
- Click **Create Web Service**
- Render will:
  - Install dependencies from `requirements.txt`
  - Run: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2`
  - Deploy on port assigned by Render

### 4. Verify Deployment
- Check build logs for errors
- Visit your Render URL: `https://candlestick-screener-XXXX.onrender.com`
- Test scanner dropdowns and date filtering

## Current Configuration (render.yaml)
```yaml
services:
  - type: web
    name: candlestick-screener
    env: python
    plan: free
    region: oregon
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2
```

## Auto-Deploy from GitHub
Every push to `master` branch automatically triggers Render deployment.

## Troubleshooting

### Database Connection Error
- Verify `DUCKDB_PATH` points to MotherDuck or external database
- Local file paths won't persist on Render's ephemeral storage

### Port Binding Error
- App now reads `PORT` from environment (Render provides this)
- Default fallback: 8000 for local development

### Build Timeout
- Free tier has limited build time
- If TA-Lib installation fails, consider prebuilt wheels or Docker

### Memory Issues
- Free tier: 512MB RAM, 0.1 CPU
- Reduce `gunicorn` workers if needed: `--workers 1`

## Features Deployed
✅ Database-driven scanner results (no on-fly calculations)
✅ Auto-submit pattern/date dropdowns
✅ Date filter with setup counts per date
✅ Loading spinner overlay
✅ Environment-aware port configuration
✅ Production-ready gunicorn server

## Next Steps
1. Monitor first deployment in Render logs
2. Set up custom domain (optional)
3. Configure alerts for downtime
4. Review performance on free tier
