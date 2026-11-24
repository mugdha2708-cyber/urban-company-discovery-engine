# Urban Company – AI Customer Discovery Engine (MVP)

A small Streamlit prototype for evidence-first classification of public customer feedback.

## What it does
- Classifies one review or a CSV batch (up to 10 rows)
- Separates sentiment from star rating
- Distinguishes repeat attempts, package visits, redo/warranty and explicit completed-repeat statements
- Preserves Unknown/Unclear where evidence is missing
- Exports JSON or CSV results

## Important limits
This tool does not access Urban Company internal systems and cannot verify booking completion,
customer identity, first-time status, or actual 90-day repeat outcomes. Outputs are research
hypotheses requiring human review. Do not upload personal or sensitive information.

## Deploy publicly with Streamlit Community Cloud
1. Create/sign in to GitHub.
2. Create a **public** repository, e.g. `urban-company-discovery-engine`.
3. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
4. Open https://share.streamlit.io and sign in with GitHub.
5. Choose **Create app**, select the repository, branch `main`, and entrypoint `app.py`.
6. In **Advanced settings → Secrets**, paste:
   ```toml
   OPENAI_API_KEY = "YOUR_API_KEY"
   OPENAI_MODEL = "gpt-6-luna"
   ```
7. Click Deploy. Streamlit will provide a public `*.streamlit.app` URL.
8. Test one review and a CSV of 2–3 rows. Share the URL only after successful testing.

## API key safety
- Create an API key at https://platform.openai.com/api-keys.
- Never commit the key to GitHub, app.py, README, or any public file.
- Add it only in Streamlit app Secrets.
- API usage may incur charges; set a budget/usage limit in your API account.

## CSV format
Required column: `review_text`
Optional columns: `source_id`, `source`, `rating`

Example:
```csv
source_id,source,rating,review_text
R001,Google Play,1,"The professional cancelled at the last minute."
R002,Trustpilot,5,"The technician arrived on time and fixed the AC."
```
