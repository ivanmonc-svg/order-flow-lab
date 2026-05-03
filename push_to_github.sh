#!/bin/bash
# Push order-flow-lab to GitHub
# Run from the order_flow_lab directory:
#   cd "<this directory>"
#   chmod +x push_to_github.sh && ./push_to_github.sh

set -e

REPO_URL="https://github.com/ivanmonc-svg/order-flow-lab.git"

echo "=== Initializing git repo ==="
git init
git add -A
git commit -m "feat: complete order-flow-lab — Phases 0-3

- Phase 0: Strategy spec from TikTok video analysis
- Phase 1: Full pipeline (data_loader, book, features, strategy, backtest, viz)
- Phase 2: Typed MBP-10 parser, time-based book reconstruction, absorption/sweep detection
- Phase 3: Bookmap-style heatmap dashboard with datashader, trade bubbles, CVD, signal markers
- 60 passing tests
- Railway deployment ready (Procfile, nixpacks.toml, railway.json)"

echo "=== Setting branch to main ==="
git branch -M main

echo "=== Adding remote ==="
git remote add origin "$REPO_URL"

echo "=== Pushing to GitHub ==="
git push -u origin main

echo ""
echo "=== Done! ==="
echo "Repo: https://github.com/ivanmonc-svg/order-flow-lab"
echo ""
echo "Next: deploy to Railway"
echo "  1. Go to https://railway.com/new"
echo "  2. Select 'Deploy from GitHub repo'"
echo "  3. Pick ivanmonc-svg/order-flow-lab"
echo "  4. Add env var: DATABENTO_API_KEY=<your key>"
echo "  5. Railway auto-detects nixpacks.toml and deploys"
