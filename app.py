import json
import io
import time
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Urban Company Discovery Engine", page_icon="🔎", layout="wide")

SYSTEM_PROMPT = """
You are an evidence-first classifier for public customer feedback about Urban Company.
Classify only what the supplied text supports. Never infer customer identity, first-time status,
completed booking, actual repeat outcome, or a 90-day cohort outcome.

Return one json object with these keys. The required output format is json:
primary_theme, text_sentiment, service_category, journey_stage, repeat_signal,
alternative_provider_signal, assignment_scope, confidence, supporting_excerpt, caution.

Allowed sentiment: Positive, Negative, Mixed, Neutral, Unclear.
Service category: use an explicitly stated service category, otherwise Unknown.
Journey stage: Discovery, Booking, Fulfilment, Service delivery, Support & recovery,
Post-service, Unclear.
Repeat signal: No explicit signal; Prior use mentioned; Repeat attempt or reschedule;
Recurring package visit; Redo or warranty; Explicit new completed booking stated.
Alternative provider signal: Explicitly mentioned, Not mentioned, Unclear.
Assignment scope: Potential core-service CX signal; Exclude InstaHelp/quick housekeeping
from core metric but retain as adjacent context; Historical or outside scope; Unknown—verify.
Confidence: High, Medium, Low.

Special rules:
- A cancellation, no-show, second attempt, reschedule, package visit, redo or warranty visit
  is NOT automatically a qualifying repeat booking.
- A qualifying repeat requires explicit evidence of a later completed new booking within 90 days
  after a completed core-category booking. If not stated, say it is not evidenced.
- Exclude InstaHelp / quick housekeeping from the core repeat metric.
- Do not confuse professional refusal with customer cancellation.
- Determine sentiment from the written text, not star rating.
- Do not infer switching from a recommendation or mention of another provider.
- Preserve uncertainty for short, translated, corrupted or ambiguous text.
- Quote a short exact excerpt that supports the classification.
"""

st.title("🔎 Urban Company – AI Customer Discovery Engine")
st.caption("Research prototype | Public feedback classification | Evidence-first, not a repeat-rate calculator")

with st.expander("What this prototype does and does not do", expanded=False):
    st.markdown("""
    **Does:** classify supplied public review text into themes, sentiment, category, journey stage,
    repeat signal, alternative-provider signal, scope and confidence.

    **Does not:** access Urban Company systems, verify booking completion, identify customers,
    calculate the actual 90-day repeat rate, or prove causal drivers. Results are hypotheses for discovery.
    """)

def get_secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

api_key = get_secret("OPENAI_API_KEY", "")
model_name = get_secret("OPENAI_MODEL", "gpt-6-luna")

with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("Model", value=model_name, help="Must be enabled for your OpenAI API project.")
    st.caption("The API key is read from Streamlit Secrets. Never paste it into this page or a public GitHub file.")
    st.markdown("[OpenAI API keys](https://platform.openai.com/api-keys)")
    st.markdown("[Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)")

if not api_key:
    st.warning("The app is ready, but the owner must add OPENAI_API_KEY in Streamlit app Secrets before AI classification can run.")
    st.code('OPENAI_API_KEY = "your-api-key"\nOPENAI_MODEL = "gpt-6-luna"', language="toml")
    st.stop()

client = OpenAI(api_key=api_key)

def classify_one(review_text, rating=None, source="User supplied"):
    user_payload = {
        "source": source,
        "star_rating": rating,
        "review_text": review_text
    }
    response = client.responses.create(
        model=model_name,
        instructions=SYSTEM_PROMPT,
        input=json.dumps(user_payload, ensure_ascii=False),
        text={"format": {"type": "json_object"}},
        max_output_tokens=500,
    )
    raw = response.output_text
    try:
        return json.loads(raw)
    except Exception:
        return {
            "primary_theme": "Unclear",
            "text_sentiment": "Unclear",
            "service_category": "Unknown",
            "journey_stage": "Unclear",
            "repeat_signal": "No explicit signal",
            "alternative_provider_signal": "Unclear",
            "assignment_scope": "Unknown—verify",
            "confidence": "Low",
            "supporting_excerpt": "",
            "caution": "Model output was not valid JSON; manual review required.",
            "raw_output": raw
        }

tab1, tab2, tab3 = st.tabs(["Single review", "Batch CSV", "Method & guardrails"])

with tab1:
    st.subheader("Classify one review")
    review = st.text_area("Paste public review text", height=180, placeholder="Paste one review or public comment...")
    rating = st.selectbox("Star rating (optional context only)", ["Not supplied", "1", "2", "3", "4", "5"])
    if st.button("Classify review", type="primary", disabled=not review.strip()):
        with st.spinner("Classifying..."):
            result = classify_one(review.strip(), None if rating == "Not supplied" else int(rating))
        st.json(result)
        st.download_button("Download this classification (JSON)", json.dumps(result, ensure_ascii=False, indent=2),
                           file_name="uc_review_classification.json", mime="application/json")

with tab2:
    st.subheader("Classify a small CSV batch")
    st.markdown("Upload a CSV with a `review_text` column. Optional columns: `source_id`, `source`, `rating`.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded:
        try:
            input_df = pd.read_csv(uploaded).fillna("")
            if "review_text" not in input_df.columns:
                st.error("CSV must contain a column named `review_text`.")
            elif len(input_df) > 10:
                st.error("For this public prototype, upload a maximum of 10 rows per run.")
            else:
                st.write(f"{len(input_df)} records loaded.")
                st.dataframe(input_df.head(10), use_container_width=True)
                if st.button("Classify batch", type="primary"):
                    outputs = []
                    progress = st.progress(0)
                    for idx, row in input_df.iterrows():
                        text_value = str(row.get("review_text", "")).strip()
                        if not text_value:
                            outputs.append({"source_id": row.get("source_id", idx), "error": "Empty review text"})
                        else:
                            try:
                                out = classify_one(
                                    text_value,
                                    int(row["rating"]) if str(row.get("rating", "")).isdigit() else None,
                                    str(row.get("source", "User supplied"))
                                )
                                out["source_id"] = row.get("source_id", idx)
                                outputs.append(out)
                            except Exception as exc:
                                outputs.append({"source_id": row.get("source_id", idx), "error": str(exc)[:300]})
                        progress.progress((idx + 1) / len(input_df))
                    result_df = pd.DataFrame(outputs)
                    st.success("Batch run finished. Review low-confidence and error rows before using results.")
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button("Download classifications (CSV)", result_df.to_csv(index=False).encode("utf-8-sig"),
                                       file_name="uc_discovery_classifications.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

with tab3:
    st.subheader("Research guardrails")
    st.markdown("""
    - Public feedback is self-selected and cannot represent all customers.
    - Public text cannot verify customer identity, booking completion, first-time status or 90-day outcomes.
    - Do not count cancellations, attempts, package visits, redos, warranty claims or refunded bookings as qualifying repeats.
    - InstaHelp / quick housekeeping is adjacent CX context, excluded from the core metric.
    - Sentiment is derived from text independently of star rating.
    - Keep source IDs and URLs in your research evidence log; do not upload personal or sensitive data.
    - Human review is required before treating any label as validated evidence.
    """)
    st.info("This prototype does not retain an evidence database. Download results and store them in your controlled evidence log.")

st.divider()
st.caption("Urban Company fellowship research prototype • Not affiliated with or endorsed by Urban Company.")
