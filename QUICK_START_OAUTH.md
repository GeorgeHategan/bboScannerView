# 🚀 Quick Start: Google OAuth Setup

## TL;DR - What You Need To Do

### 1️⃣ Get Google Credentials (5 minutes)
https://console.cloud.google.com/apis/credentials
- Create OAuth client ID → Web application
- Redirect URI: `http://localhost:8000/auth/google/callback`
- Copy Client ID & Secret

### 2️⃣ Update .env File (1 minute)
```bash
GOOGLE_CLIENT_ID=paste-your-client-id-here
GOOGLE_CLIENT_SECRET=paste-your-secret-here
SECRET_KEY=e90543b677763aa0dee1ed7cc2e36b757a4573d973828f09d80256eb9490ebc5
ALLOWED_EMAILS=your-email@gmail.com
MOTHERDUCK_TOKEN=your-existing-token
```

### 3️⃣ Restart Server (30 seconds)
```bash
kill $(cat server.pid) 2>/dev/null || true
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4️⃣ Test (1 minute)
- Go to http://localhost:8000/login
- Click "Sign in with Google"
- Authenticate
- ✅ Should see scanner dashboard

## 📋 Pre-Generated Values

**SECRET_KEY** (already generated):
```
e90543b677763aa0dee1ed7cc2e36b757a4573d973828f09d80256eb9490ebc5
```

**OAuth Redirect URI** (use exactly this):
```
http://localhost:8000/auth/google/callback
```

## ⚠️ Common Mistakes

❌ **Don't**: Add quotes around values in .env
✅ **Do**: `GOOGLE_CLIENT_ID=123456.apps.googleusercontent.com`

❌ **Don't**: Forget to add your email to ALLOWED_EMAILS
✅ **Do**: `ALLOWED_EMAILS=you@gmail.com,friend@gmail.com`

❌ **Don't**: Use trailing slash in redirect URI
✅ **Do**: `http://localhost:8000/auth/google/callback`

❌ **Don't**: Commit .env file to git
✅ **Do**: Keep .env in .gitignore (it already is)

## 🆘 Troubleshooting

**"redirect_uri_mismatch"**
→ Copy exactly: `http://localhost:8000/auth/google/callback`

**"Access denied"**
→ Add your email to ALLOWED_EMAILS in .env

**"Client secret not configured"**
→ Check GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env

**Server won't start**
→ Verify all environment variables are set

## 📚 Full Documentation

- **Setup Guide**: `GOOGLE_OAUTH_SETUP.md`
- **Implementation Details**: `OAUTH_IMPLEMENTATION_SUMMARY.md`
- **Environment Template**: `.env.example`

## ✅ Success Indicators

You'll know it's working when:
- ✅ Login page shows "Sign in with Google" button
- ✅ Clicking button redirects to Google
- ✅ After Google auth, redirects to scanner dashboard
- ✅ Can access all pages (stats, scanner-docs, etc.)
- ✅ Logout works and requires re-login

## 🎯 Your To-Do List

```markdown
- [ ] Go to https://console.cloud.google.com/apis/credentials
- [ ] Create OAuth 2.0 Client ID
- [ ] Copy Client ID and Client Secret
- [ ] Paste into .env file
- [ ] Add your email to ALLOWED_EMAILS in .env
- [ ] Add SECRET_KEY (already generated above)
- [ ] Restart server
- [ ] Test login
```

**Estimated Time**: 10 minutes total

---

Need help? Check `GOOGLE_OAUTH_SETUP.md` for detailed instructions.
