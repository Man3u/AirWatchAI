# Deploying AirWatchAI to Streamlit Community Cloud

Repo is deploy-ready: `app.py` at the root, `requirements.txt` pinned,
`.streamlit/config.toml` set, and every file the app needs at runtime
(`outputs/aq_combined_weekly.csv`, `models/normalization_stats.json`,
`models/no2_forecaster_weights.npz`, `models/city_skill_summary.json`)
is already committed. Total repo size is ~2.6MB, well under any limit.

These last two steps need your own GitHub and Streamlit accounts, so
they're for you to click through -- here's exactly what to do.

## 1. Push this repo to GitHub

1. Go to github.com -> New repository -> name it `AirWatchAI` (public,
   so it's visible in your portfolio) -> **do not** initialize with a
   README/gitignore (this repo already has both) -> Create repository.
2. Copy the repo URL GitHub shows you (looks like
   `https://github.com/<your-username>/AirWatchAI.git`).
3. Run these two commands in this project folder:
   ```
   git remote add origin https://github.com/<your-username>/AirWatchAI.git
   git push -u origin master
   ```
   (Or paste that URL back and I'll run the push for you.)

## 2. Deploy on Streamlit Community Cloud

1. Go to share.streamlit.io -> sign in with your GitHub account (this
   is what actually links the two -- no separate password to manage).
2. Click "New app".
3. Repository: `<your-username>/AirWatchAI`. Branch: `master`.
   Main file path: `app.py`.
4. Click "Deploy". First deploy takes 1-2 minutes to install the 5
   packages in `requirements.txt` (no PyTorch, so this is fast).
5. You get a URL like `https://airwatchai-<random>.streamlit.app` --
   that's the link to put in your portfolio/resume/LinkedIn.

## Updating the live app later

Any `git push` to the connected branch auto-redeploys within
~30 seconds -- no redeploy button to remember.
