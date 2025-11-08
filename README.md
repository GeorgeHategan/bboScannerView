# candlestick-screener
web-based technical screener for candlestick patterns using TA-Lib, Python, and Flask

## Features
- Multiple candlestick pattern detection (TA-Lib patterns + custom patterns)
- Volume analysis with 20-day average comparison
- Earnings date tracking with countdown
- News sentiment analysis (Alpha Vantage)
- Market cap and sector information
- Multiple scanner types: Qullamaggie, Momentum Burst, Supertrend, Explosive Volume
- Pattern strength scoring and quality assessment
- TradingView chart integration

## Video Tutorials for this repository:

* Candlestick Pattern Recognition - https://www.youtube.com/watch?v=QGkf2-caXmc
* Building a Web-based Technical Screener - https://www.youtube.com/watch?v=OhvQN_yIgCo
* Finding Breakouts - https://www.youtube.com/watch?v=exGuyBnhN_8

## Deploy to Render (Free Hosting)

### Prerequisites
1. A MotherDuck (or DuckDB) database with your stock data
2. GitHub account
3. Render account (free) - https://render.com

### Step-by-Step Deployment:

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin master
   ```

2. **Sign up for Render:**
   - Go to https://render.com
   - Sign up with your GitHub account

3. **Create a New Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub account if not already connected
   - Select your `candlestick-screener` repository

4. **Configure the Service:**
   - **Name**: candlestick-screener (or any name you want)
   - **Region**: Choose closest to you
   - **Branch**: master
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Plan**: Free

5. **Set Environment Variables:**
   Click "Advanced" → Add Environment Variables:
   
   - **DUCKDB_PATH**: Your MotherDuck connection string
     ```
     md:?motherduck_token=YOUR_TOKEN
     ```
     Or if using a remote DuckDB file, provide the URL
   
   - **ALPHA_VANTAGE_API_KEY**: Your Alpha Vantage API key
     ```
     75IGYUZ3C7AC2PBM
     ```

6. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Wait 5-10 minutes for the first deployment

7. **Access Your App:**
   - Once deployed, you'll get a URL like: `https://candlestick-screener.onrender.com`
   - Your app is now live!

### Important Notes:

- **Free Tier Limitations**: 
  - App spins down after 15 minutes of inactivity
  - First request after spin-down takes ~30 seconds
  - 750 hours/month free (enough for most use cases)

- **MotherDuck Setup**:
  Your DuckDB connection string should look like:
  ```
  md:?motherduck_token=your_motherduck_token_here
  ```
  Get your token from https://app.motherduck.com/

- **Keeping App Awake** (Optional):
  Use a service like UptimeRobot to ping your app every 14 minutes to prevent spin-down

### Troubleshooting:

- **Build fails on TA-Lib**: Render should have TA-Lib pre-installed, but if it fails, you may need to use a Docker deployment instead
- **Database connection issues**: Verify your DUCKDB_PATH environment variable is correct
- **Check logs**: In Render dashboard, go to your service → Logs to see error messages