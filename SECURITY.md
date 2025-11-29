# Security Guidelines

## 🔐 API Key Security

### ⚠️ IMPORTANT: Never commit API keys to version control!

This project uses a **per-user API key system** where users provide their own keys through the UI. This ensures:
- **Privacy**: Each user's keys stay in their session only
- **Cost Control**: Users pay for their own API usage  
- **Security**: No shared credentials or server-side storage

### 🛡️ Security Best Practices

#### For Development:
1. **Use .env.example**: Copy to `.env` and add your keys for local testing
2. **Never commit .env**: The `.gitignore` file prevents this
3. **Rotate keys regularly**: Generate new API keys periodically
4. **Use environment variables**: Never hardcode keys in source code

#### For Production:
1. **Users provide keys**: No server-side API key storage required
2. **Session-only storage**: Keys stored in `st.session_state` only
3. **No persistence**: Keys are never written to files or databases
4. **Clear warnings**: UI clearly states keys are session-only

### 🔧 Local Development Setup

1. Copy the example file:
```bash
cp .env.example .env
```

2. Add your API keys to `.env`:
```bash
OPENAI_API_KEY=your_actual_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```

3. The `.gitignore` file ensures `.env` is never committed

### 🚀 Deployment Security

#### Streamlit Cloud:
- No server-side secrets needed
- Users enter keys in UI
- Automatic session cleanup

#### Self-Hosted:
- Use environment variables for any fallback keys
- Implement rate limiting
- Monitor API usage
- Set up proper logging (without logging keys)

### 🔍 Security Checklist

- [ ] `.env` file is in `.gitignore`
- [ ] No API keys in source code
- [ ] Users provide their own keys via UI
- [ ] Keys are session-only (not persisted)
- [ ] Clear privacy notices in UI
- [ ] Regular security audits

### 📞 Reporting Security Issues

If you find a security vulnerability, please report it responsibly:
1. Do not create a public GitHub issue
2. Contact the developer directly
3. Provide detailed information about the vulnerability
4. Allow time for the issue to be addressed before public disclosure

---

**Remember**: The best security practice is defense in depth. This project implements multiple layers of security to protect user credentials and ensure safe operation.