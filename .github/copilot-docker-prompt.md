# GitHub Copilot Prompt: Docker Development Environment Troubleshooting

## Copy and paste this prompt into GitHub Copilot Chat for expert Docker troubleshooting assistance:

---

**I'm working with a Brain-Computer Interface (BCI) data compression toolkit that uses a comprehensive Docker development environment. The setup includes:**

**Architecture:**
- Multi-container stack: Backend (Python/PyTorch), Frontend (React/Vite), PostgreSQL, Redis, MongoDB, MinIO
- VS Code Dev Container integration with 20+ extensions
- GPU acceleration support for neural data processing
- Real-time processing requirements (<1ms latency for BCI applications)

**Key Files:**
- `docker-compose.dev.yml` (157 lines, 6 services with health checks)
- `docker/dev-backend.Dockerfile` (Python with NumPy, SciPy, PyTorch, CuPy)
- `docker/dev-frontend.Dockerfile` (Node.js with global packages)
- `.devcontainer/devcontainer.json` (140 lines with comprehensive VS Code setup)

**Common Issues I Need Help With:**

1. **Frontend Build Failures**: npm permission errors when installing global packages (yarn, pnpm, vite) in container
2. **Port Conflicts**: Services failing to start due to port allocation conflicts (8888, 3000, 5432, etc.)
3. **Volume Permission Issues**: Cannot write files in mounted volumes, files owned by root instead of user
4. **Module Import Errors**: Python packages not found (torch, numpy, bci_compression package)
5. **VS Code Dev Container Problems**: Container fails to start, extensions not loading, Python interpreter not found
6. **Database Connection Issues**: Services can't connect to postgres/redis/mongodb containers
7. **GPU Access Problems**: CUDA not available in containers, GPU acceleration failing
8. **Performance Issues**: Slow builds, high memory usage, container startup delays

**Expected Behavior:**
- All 6 services start successfully with health checks passing
- VS Code Dev Container opens with full Python/Jupyter support
- Neural compression algorithms run with GPU acceleration
- Real-time data processing with <1ms latency
- Seamless development workflow with hot-reload

**Current Environment:**
- OS: Linux/Windows/macOS
- Docker version: [specify your version]
- Docker Compose version: [specify your version]
- Available RAM: [specify]
- GPU: [specify if applicable]

**Specific Problem:** [Describe your current issue here]

**Error Messages:** [Include any error messages you're seeing]

**What I've Tried:** [List troubleshooting steps you've already attempted]

**Please provide:**
1. **Root cause analysis** of the issue
2. **Step-by-step solution** with exact commands
3. **Prevention strategies** to avoid this issue in future
4. **Alternative approaches** if the primary solution doesn't work
5. **Validation steps** to confirm the fix worked

**Context Notes:**
- This is for BCI research with neural data processing
- Multi-channel recordings (32-256+ channels) at high frequencies
- Need consistent cross-platform development environment
- Real-time processing requirements for brain-computer interfaces

---

## Alternative Shorter Prompt (for quick issues):

**Docker BCI Development Environment Issue**

I have a multi-container Docker setup (Python backend, React frontend, PostgreSQL/Redis/MongoDB) for brain-computer interface development.

**Issue:** [Describe your problem]
**Error:** [Include error message]
**Tried:** [What you've attempted]

Need: root cause, exact fix commands, and validation steps. This is for real-time neural data processing with <1ms latency requirements.

---

## Usage Instructions:

1. **Copy the appropriate prompt** (full or shorter version)
2. **Fill in the specific details** in brackets with your actual information
3. **Paste into GitHub Copilot Chat** or Copilot inline chat
4. **Follow the provided solution** step by step
5. **Run validation tests** to confirm the fix

## Pro Tips for Better Copilot Responses:

- **Include exact error messages** - copy/paste the full error output
- **Specify your OS and Docker versions** - compatibility matters
- **Mention what you've already tried** - avoids redundant suggestions
- **Describe the expected behavior** - helps Copilot understand the goal
- **Include relevant file contents** - paste configuration files if needed
- **Ask for validation steps** - ensure the fix actually works

## Follow-up Questions to Ask:

After getting a solution, you can ask:
- "How can I prevent this issue in the future?"
- "What are the performance implications of this fix?"
- "Is there a more optimal approach for BCI applications?"
- "How do I monitor this to catch issues early?"
- "Can you help me create a test to verify this works?"

This prompt framework will help you get more targeted, actionable solutions from GitHub Copilot for your Docker development environment issues.
