# Public Repository Submission Checklist

- [x] Public-repository-safe `.gitignore`
- [x] `.env` and `.env.*` excluded
- [x] `.env.example` included with placeholders
- [x] Local uploads/reference images excluded
- [x] Local SQLite database excluded
- [x] Model weights excluded
- [x] README with setup instructions included
- [x] Dependency list included
- [x] Original application source retained as `app.py`
- [x] Python syntax check passed on the supplied source

Before publishing:
1. Run `git status` and confirm no secret or local-data files are staged.
2. Search the repository for API keys, passwords, tokens, and service-account files.
3. If a real credential was ever committed, revoke/rotate it before public submission.
