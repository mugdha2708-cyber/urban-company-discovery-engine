import json
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Urban Company Discovery Engine", page_icon="🔎", layout="wide")
st.title("🔎 Urban Company – AI Customer Discovery Engine")
st.caption("Evidence-first research prototype | Public feedback | Exploratory opportunity analysis")

SYSTEM_PROMPT = """
You are an evidence-first classifier for public customer feedback about Urban Company.
Return exactly one JSON object with keys:
primary_theme, text_sentiment, service_category, journey_stage, repeat_signal,
alternative_provider_signal, assignment_scope, confidence, supporting_excerpt, caution.
Sentiment: Positive, Negative, Mixed, Neutral, Unclear.
Journey: Discovery, Booking, Fulfilment, Service delivery, Support & recovery, Post-service, Unclear.
Repeat: No explicit signal; Prior use mentioned; Repeat attempt or reschedule; Recurring package visit;
Redo or warranty; Explicit new completed booking stated.
Alternative provider: Explicitly mentioned, Not mentioned, Unclear.
Scope: Potential core-service CX signal; Exclude InstaHelp/quick housekeeping from core metric but retain as adjacent context;
Historical or outside scope; Unknown—verify.
Confidence: High, Medium, Low.
Rules:
- Do not infer identity, first-time status, completed booking, actual repeat outcome, or 90-day cohort outcome.
- A cancellation, no-show, reschedule, package visit, redo or warranty visit is NOT automatically a qualifying repeat.
- Qualifying repeat requires evidence of a later completed new booking within 90 days after a completed core-category booking.
- Exclude InstaHelp / quick housekeeping from the core repeat metric.
- Sentiment comes from text, not star rating.
- Do not infer switching from a recommendation or mention of another provider.
- "Regular technician" or "another technician" does not establish provider affiliation; use Unclear unless explicit.
- Preserve ambiguity and quote a short exact supporting excerpt.
"""

def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("OpenAI model", value=secret("OPENAI_MODEL", "gpt-6-luna"))
    st.caption("API key is read only from Streamlit Secrets. Never paste it into this page or GitHub.")
    st.markdown("[OpenAI API keys](https://platform.openai.com/api-keys)")
    st.markdown("[Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)")

api_key = secret("OPENAI_API_KEY")
if not api_key:
    st.warning("Owner setup required: add OPENAI_API_KEY to Streamlit app Secrets to enable AI classification.")
    st.code('OPENAI_API_KEY = "your-api-key"\nOPENAI_MODEL = "gpt-6-luna"', language="toml")
    st.stop()

client = OpenAI(api_key=api_key)

def classify_one(review_text, rating=None, source="User supplied"):
    payload = {"source": source, "star_rating_context_only": rating, "review_text": review_text}
    response = client.responses.create(
        model=model_name,
        instructions=SYSTEM_PROMPT,
        input="Classify the supplied customer feedback and return the result as a valid JSON object matching the required schema.\n" + json.dumps(payload, ensure_ascii=False),
        text={"format": {"type": "json_object"}},
        max_output_tokens=550,
    )
    return json.loads(response.output_text)

def add_context_columns(df):
    defaults = {
        "city": "Unknown", "household_type": "Unknown",
        "acquisition_channel": "Unknown", "service_category_context": "Unknown"
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna("").astype(str).replace("", default)
    return df

def opportunity_table(results):
    if results.empty or "primary_theme" not in results:
        return pd.DataFrame()
    x = results.copy()
    x["primary_theme"] = x["primary_theme"].fillna("Unclear")
    counts = x.groupby("primary_theme").size().rename("Records").reset_index()
    counts["Share of submitted sample (%)"] = (counts["Records"] / len(x) * 100).round(1)
    if "confidence" in x:
        low = x.assign(is_low=x["confidence"].isin(["Low", "Medium"])).groupby("primary_theme")["is_low"].sum()
        counts["Low/medium confidence records"] = counts["primary_theme"].map(low).fillna(0).astype(int)
    counts["Interpretation"] = "Exploratory frequency only; not population prevalence or causal impact"
    return counts.sort_values("Records", ascending=False)

def segment_coverage(df):
    rows = []
    for col, label in [
        ("service_category_context", "Service category"),
        ("city", "City"),
        ("household_type", "Household type"),
        ("acquisition_channel", "Acquisition channel"),
    ]:
        vals = df[col].replace("", "Unknown")
        known = int((vals.str.lower() != "unknown").sum())
        rows.append({"Dimension": label, "Records with known value": known,
                     "Unknown / missing": int(len(df)-known),
                     "Known coverage (%)": round(known/len(df)*100,1) if len(df) else 0})
    return pd.DataFrame(rows)

tabs = st.tabs(["Single review", "Batch + segments", "Opportunity comparison", "Bias & evidence limits"])
with tabs[0]:
    st.subheader("Classify one public review")
    review = st.text_area("Review text", height=170, placeholder="Paste a public review or comment…")
    rating = st.selectbox("Star rating (optional; context only)", ["Not supplied","1","2","3","4","5"])
    source = st.text_input("Source label (optional)", "User supplied")
    if st.button("Classify review", type="primary", disabled=not review.strip()):
        try:
            with st.spinner("Classifying evidence…"):
                result = classify_one(review.strip(), None if rating=="Not supplied" else int(rating), source)
            st.json(result)
            st.download_button("Download JSON", json.dumps(result, ensure_ascii=False, indent=2),
                               "uc_review_classification.json", "application/json")
        except Exception as e:
            st.error(f"Classification failed: {str(e)[:400]}")

with tabs[1]:
    st.subheader("Batch classification with optional segmentation")
    st.markdown("Upload CSV with `review_text`. Optional columns: `source_id`, `source`, `rating`, `city`, `household_type`, `acquisition_channel`, `service_category_context`.")
    st.caption("Only enter segmentation attributes explicitly available from permitted public evidence or valid research data. Do not infer them.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded:
        try:
            source_df = add_context_columns(pd.read_csv(uploaded).fillna(""))
            if "review_text" not in source_df:
                st.error("CSV needs a review_text column.")
            elif len(source_df) > 10:
                st.error("Maximum 10 records per run in this prototype.")
            else:
                st.dataframe(source_df, use_container_width=True)
                if st.button("Classify batch", type="primary"):
                    outputs=[]
                    progress=st.progress(0)
                    for i,row in source_df.iterrows():
                        try:
                            if not str(row.get("review_text","")).strip():
                                raise ValueError("Empty review text")
                            out=classify_one(str(row["review_text"]), int(row["rating"]) if str(row.get("rating","")).isdigit() else None, str(row.get("source","User supplied")))
                            out["source_id"]=row.get("source_id",i)
                            for c in ["city","household_type","acquisition_channel","service_category_context"]:
                                out[c]=row.get(c,"Unknown")
                            outputs.append(out)
                        except Exception as e:
                            outputs.append({"source_id":row.get("source_id",i),"error":str(e)[:250]})
                        progress.progress((i+1)/len(source_df))
                    result_df=pd.DataFrame(outputs)
                    st.session_state["last_results"]=result_df
                    st.session_state["last_input"]=source_df
                    st.success("Batch complete. Inspect low-confidence and error rows.")
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button("Download classified CSV", result_df.to_csv(index=False).encode("utf-8-sig"),
                                       "uc_discovery_classifications.csv","text/csv")
                    st.subheader("Segmentation data coverage")
                    st.dataframe(segment_coverage(source_df), use_container_width=True)
        except Exception as e:
            st.error(f"Could not process CSV: {str(e)[:300]}")

with tabs[2]:
    st.subheader("Opportunity comparison")
    st.markdown("Compares themes in the currently submitted batch. Frequency is descriptive only; it is not a customer-level repeat-rate estimate.")
    res=st.session_state.get("last_results")
    if res is not None:
        table=opportunity_table(res)
        if not table.empty:
            st.dataframe(table, use_container_width=True)
            st.bar_chart(table.set_index("primary_theme")["Records"])
            st.info("Use this as an exploratory prioritisation input alongside severity, source breadth, repeat relevance and human review. Do not interpret frequency as causal impact.")
        else:
            st.info("No usable classified results yet. Run a batch first.")
    else:
        st.info("Run a batch in the previous tab to populate this comparison.")

with tabs[3]:
    st.subheader("Bias handling and evidence limitations")
    st.markdown("""
    **Known biases**
    - **Voluntary-review / negativity bias:** people who post publicly are self-selected; negative experiences may be overrepresented.
    - **Silent non-returners:** customers who simply do not book again usually leave no public trace. Public reviews cannot reveal their reasons.
    - **Source and geography bias:** platform audiences, language, rating behaviour and reviewer location may not represent service locations or all customers.
    - **Missing segmentation:** city, household type, acquisition channel and service category are often absent. Unknown must remain unknown.
    - **AI classification error:** themes and journey stages are model interpretations; low-confidence or ambiguous records need human review.

    **Mitigation in this prototype**
    - Report source volumes and coverage separately; do not pool sources as if they were representative.
    - Preserve unknown/missing segment values; show coverage before comparing segments.
    - Use a stratified, human-labelled Golden Set and log corrections.
    - Treat public feedback as directional hypothesis generation, not proof of return/non-return behaviour.
    - Validate silent non-returner explanations through interviews, surveys and (if authorised) cohort-level booking data.
    - Do not calculate actual 90-day repeat rate without customer-level cohort and completed-booking data.
    """)
    st.warning("No statistical reweighting is applied. This prototype documents and limits interpretation of bias; it does not eliminate bias.")

st.divider()
st.caption("Urban Company fellowship research prototype • Not affiliated with or endorsed by Urban Company.")
