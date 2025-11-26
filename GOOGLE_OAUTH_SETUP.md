# Google OAuth Setup Guide

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google+ API** or **People API**:
   - Navigate to **APIs & Services** > **Library**
   - Search for "Google+ API" or "People API"
   - Click **Enable**

## Step 2: Create OAuth 2.0 Credentials

1. Go to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **OAuth client ID**
3. Configure the OAuth consent screen if prompted:
   - User Type: **External** (or Internal if using Google Workspace)
   - App name: **BBO Scanner**
   - User support email: Your email
   - Developer contact: Your email
   - Scopes: Add `openid`, `email`, `profile`
4. Application type: **Web application**
5. Name: **BBO Scanner Web Client**
6. Authorized redirect URIs:
   - For local development: `http://localhost:8000/auth/google/callback`
   - For production: `https://yourdomain.com/auth/google/callback`
7. Click **Create**
8. Copy the **Client ID** and **Client Secret**

## Step 3: Configure Environment Variables

Add these to your `.env` file (DO NOT commit this file to git):

```bash
# Google OAuth Credentials (from Google Cloud Console)
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret

# Session Secret Key (already generated for you)
SECRET_KEY=e90543b677763aa0dee1ed7cc2e36b757a4573d973828f09d80256eb9490ebc5

# Allowed Users (comma-separated email addresses)
ALLOWED_EMAILS=your-email@gmail.com,another-user@gmail.com

# Existing MotherDuck Token
MOTHERDUCK_TOKEN=your-existing-token
```

## Step 4: Update Your .env File

1. Make sure `.env` is in `.gitignore` (it should be)
2. Add the environment variables above to your `.env` file
3. Replace placeholders with actual values:
   - `GOOGLE_CLIENT_ID`: From Step 2
   - `GOOGLE_CLIENT_SECRET`: From Step 2
   - `ALLOWED_EMAILS`: Your authorized users' email addresses
   - `MOTHERDUCK_TOKEN`: Your existing token (should already be set)

## Step 5: Restart Server

```bash
# Kill existing server if running
kill $(cat server.pid) 2>/dev/null || true

# Start server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Step 6: Test OAuth Flow

1. Open browser to http://localhost:8000/login
2. Click "Sign in with Google"
3. Authenticate with Google account
4. Verify your email is in ALLOWED_EMAILS list
5. You should be redirected to the main dashboard

## Troubleshooting

### "redirect_uri_mismatch" error
- Check that the redirect URI in Google Console exactly matches: `http://localhost:8000/auth/google/callback`
- Make sure there are no trailing slashes

### "Access denied" after successful Google login
- Verify your email is in the ALLOWED_EMAILS environment variable
- Check for typos in email addresses
- Restart the server after changing .env

### "Client secret not configured"
- Make sure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are set in .env
- Verify python-dotenv is loading the .env file
- Check that .env is in the same directory as app.py

### Server won't start
- Verify all environment variables are set
- Check SECRET_KEY is configured
- Look for syntax errors in .env file (no quotes needed around values)

## Security Notes

- **Never commit .env file** - it contains secrets
- Use different OAuth credentials for development and production
- Rotate SECRET_KEY periodically
- Keep ALLOWED_EMAILS list up to date
- Use HTTPS in production (required by Google for OAuth)

## Production Deployment

For production (e.g., Render, Heroku):

1. Add environment variables in your hosting platform's dashboard
2. Update authorized redirect URIs to use your production domain
3. Use HTTPS for all OAuth callbacks
4. Consider using a secrets manager instead of .env file
