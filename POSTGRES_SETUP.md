# PostgreSQL Setup on Render

Your user management has been migrated to use PostgreSQL instead of SQLite. This ensures user data persists across deployments on Render.

## Setup Steps

### 1. Create PostgreSQL Database on Render

1. Go to https://dashboard.render.com
2. Click "New +" → "PostgreSQL"
3. Configure:
   - **Name**: `bboScanner-users` (or any name you want)
   - **Database**: `users_db`
   - **User**: (auto-generated)
   - **Region**: Same as your web service
   - **Instance Type**: **Free** (sufficient for user management)
4. Click "Create Database"
5. Wait for it to provision (~2 minutes)

### 2. Get the Connection String

1. Click on your new PostgreSQL database
2. Find the "Internal Database URL" or "External Database URL"
3. It will look like: `postgres://user:pass@host/database`
4. Copy this URL

### 3. Add Environment Variable to Your Web Service

1. Go to your web service (bboScannerView)
2. Click "Environment" in the left sidebar
3. Add new environment variable:
   - **Key**: `DATABASE_URL`
   - **Value**: (paste the PostgreSQL URL you copied)
4. Click "Save Changes"

### 4. Deploy

Render will automatically redeploy with the new DATABASE_URL.

On startup, the app will:
- Detect DATABASE_URL is set
- Connect to PostgreSQL instead of SQLite
- Create the required tables automatically
- Add hategan@gmail.com as admin
- Your user management is now persistent! 🎉

## Current Allowed Users

After deployment, these users will be in the database:
- hategan@gmail.com (admin)
- stubberup@gmail.com

## Adding More Users

Use the admin panel at: https://your-app.onrender.com/admin

## Migration (Optional)

If you had users in SQLite that you want to migrate to PostgreSQL:

```bash
# Set DATABASE_URL locally (from Render dashboard)
export DATABASE_URL="your-postgres-url-here"

# Run migration
python3 migrate_users_to_postgres.py
```

## How It Works

- **Production (Render)**: Uses PostgreSQL via DATABASE_URL
- **Local Development**: Uses SQLite (users.db file)
- **Code**: Automatically detects which database to use

## Benefits

✅ User data persists across Render deployments  
✅ Login logs persist  
✅ No data loss on redeploy  
✅ Free tier PostgreSQL on Render  
✅ Better performance for relational data  
