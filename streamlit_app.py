
import re
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from googleapiclient.discovery import build
from dateutil import parser as dateparser

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

st.set_page_config(page_title="YouTube Analytics - Capstone", page_icon="📊", layout="wide")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def get_youtube_client():
    api_key = st.secrets.get("YOUTUBE_API_KEY", "AIzaSyDHZ63uhU-GFojNCYt-nQGpVmYqbLRqxTw")
    if not api_key:
        st.error("❌ Missing YOUTUBE_API_KEY in secrets. Add it in Streamlit Cloud 'App Settings → Secrets' or in .streamlit/secrets.toml for local.")
        st.stop()
    return build("youtube", "v3", developerKey=api_key)

CHANNEL_URL_PAT = re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/(?:channel/|@|c/)?([^/?#]+)")

def parse_iso8601_duration(duration: str) -> int:
    if not isinstance(duration, str):
        return 0
    hours = minutes = seconds = 0
    match = re.match(r'^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$', duration)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds

def try_extract_channel_token(text: str) -> str:
    text = text.strip()
    if text.startswith("UC") and len(text) >= 10:
        return text
    m = CHANNEL_URL_PAT.search(text)
    if m:
        token = m.group(1)
        if token.startswith('@'):
            return token
        return token
    if text.startswith('@'):
        return text
    return text

@st.cache_data(show_spinner=False)
def resolve_channel_id(youtube, token: str) -> Tuple[str, str]:
    if token.startswith("UC"):
        resp = youtube.channels().list(part="snippet", id=token, maxResults=1).execute()
        items = resp.get("items", [])
        if items:
            return token, items[0]["snippet"]["title"]
    q = token.lstrip('@')
    resp = youtube.search().list(part="snippet", q=q, type="channel", maxResults=1).execute()
    items = resp.get("items", [])
    if not items:
        return "", ""
    ch_id = items[0]["snippet"]["channelId"]
    ch_title = items[0]["snippet"]["channelTitle"]
    return ch_id, ch_title

@st.cache_data(show_spinner=False)
def get_channel_stats(youtube, channel_ids: list) -> pd.DataFrame:
    all_rows = []
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i+50]
        resp = youtube.channels().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch),
            maxResults=50
        ).execute()
        for item in resp.get("items", []):
            stats = item.get("statistics", {})
            details = item.get("contentDetails", {})
            row = {
                "channel_id": item["id"],
                "channel_title": item["snippet"]["title"],
                "published_at": item["snippet"].get("publishedAt"),
                "uploads_playlist_id": details.get("relatedPlaylists", {}).get("uploads"),
                "subscribers": int(stats.get("subscriberCount", 0)) if stats.get("hiddenSubscriberCount") != True else np.nan,
                "views": int(stats.get("viewCount", 0)),
                "video_count": int(stats.get("videoCount", 0)),
            }
            all_rows.append(row)
    return pd.DataFrame(all_rows)

@st.cache_data(show_spinner=False)
def get_video_ids(youtube, uploads_playlist_id: str, max_videos: int = 1000) -> list:
    video_ids = []
    page_token = None
    while True and len(video_ids) < max_videos:
        resp = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=page_token
        ).execute()
        for it in resp.get("items", []):
            vid = it["contentDetails"]["videoId"]
            video_ids.append(vid)
            if len(video_ids) >= max_videos:
                break
        page_token = resp.get("nextPageToken")
        if not page_token or len(video_ids) >= max_videos:
            break
    return video_ids

@st.cache_data(show_spinner=False)
def get_video_details(youtube, video_ids: list) -> pd.DataFrame:
    records = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch),
            maxResults=50
        ).execute()
        for item in resp.get("items", []):
            sn = item.get("snippet", {})
            ct = item.get("contentDetails", {})
            stt = item.get("statistics", {})
            duration_sec = parse_iso8601_duration(ct.get("duration", ""))
            tags = sn.get("tags", [])
            rec = {
                "video_id": item["id"],
                "title": sn.get("title"),
                "description": sn.get("description"),
                "published_at": sn.get("publishedAt"),
                "duration_sec": duration_sec,
                "duration_min": round(duration_sec / 60.0, 2) if duration_sec else 0,
                "views": int(stt.get("viewCount", 0)),
                "likes": int(stt.get("likeCount", 0)) if "likeCount" in stt else np.nan,
                "comments": int(stt.get("commentCount", 0)) if "commentCount" in stt else np.nan,
                "tags": ", ".join(tags) if isinstance(tags, list) else "",
                "default_language": sn.get("defaultLanguage"),
                "category_id": sn.get("categoryId")
            }
            records.append(rec)
    df = pd.DataFrame(records)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df.sort_values("published_at", ascending=False, inplace=True)
    return df

def build_wordcloud(text: str):
    sw = set(stopwords.words("english"))
    tokens = [t.lower() for t in word_tokenize(text) if t.isalpha() and t.lower() not in sw]
    if not tokens:
        return None
    wc = WordCloud(width=1200, height=600, background_color="white").generate(" ".join(tokens))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_axis_off()
    return fig

st.title("📊 YouTube Analytics Capstone — Streamlit App")
st.write("Analyze channels and their videos using the YouTube Data API v3, NLP, and visualizations.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Enter one or more **Channel URLs / @handles / UC IDs** (comma-separated).")
    user_input = st.text_area("Channels", placeholder="e.g. https://youtube.com/@freecodecamp, UCYO_jab_esuFRV4b17AJtAw", height=80)
    max_videos = st.number_input("Max videos per channel", 50, 2000, 400, 50)
    run_btn = st.button("Fetch & Analyze", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Tip: You can also upload a CSV with a 'channel' column (URL/handle/ID).")
    file = st.file_uploader("Upload channel list (CSV)", type=["csv"])

if file is not None:
    try:
        df_up = pd.read_csv(file)
        if "channel" in df_up.columns:
            extra = ", ".join(map(str, df_up["channel"].dropna().tolist()))
            user_input = (user_input + ", " + extra) if user_input else extra
            st.success(f"Loaded {df_up.shape[0]} channels from CSV.")
        else:
            st.warning("CSV must contain a column named 'channel'.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if run_btn:
    if not user_input.strip():
        st.warning("Please provide at least one channel input.")
        st.stop()

    youtube = get_youtube_client()

    raw_tokens = [tok.strip() for tok in user_input.split(",") if tok.strip()]
    parsed = [try_extract_channel_token(tok) for tok in raw_tokens]

    resolved = []
    with st.spinner("Resolving channel IDs..."):
        for tok in parsed:
            ch_id, ch_title = resolve_channel_id(youtube, tok)
            if ch_id:
                resolved.append((ch_id, ch_title))
            else:
                st.warning(f"Could not resolve channel from: {tok}")

    if not resolved:
        st.error("No valid channels resolved. Please check your inputs.")
        st.stop()

    st.success(f"Resolved {len(resolved)} channel(s).")

    channel_ids = [cid for cid, _ in resolved]
    with st.spinner("Fetching channel statistics..."):
        ch_df = get_channel_stats(youtube, channel_ids)
    st.subheader("📺 Channel Overview")
    st.dataframe(ch_df, use_container_width=True)

    all_videos = []
    for _, row in ch_df.iterrows():
        upl = row["uploads_playlist_id"]
        if not upl:
            continue
        with st.spinner(f"Fetching videos for {row['channel_title']}..."):
            vids = get_video_ids(youtube, upl, max_videos=int(max_videos))
            vdf = get_video_details(youtube, vids)
            if not vdf.empty:
                vdf["channel_title"] = row["channel_title"]
                vdf["channel_id"] = row["channel_id"]
                all_videos.append(vdf)

    if not all_videos:
        st.warning("No videos fetched for the provided channels.")
        st.stop()

    vids_df = pd.concat(all_videos, ignore_index=True)
    st.subheader("🎬 Video Dataset")
    st.dataframe(vids_df.head(100), use_container_width=True)

    csv = vids_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv, file_name="youtube_videos.csv", mime="text/csv")

    st.subheader("📈 Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 15 Videos by Views**")
        top_views = vids_df.sort_values("views", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.barh(top_views["title"].iloc[::-1], top_views["views"].iloc[::-1])
        ax.set_xlabel("Views")
        ax.set_ylabel("Video Title")
        ax.set_title("Top 15 by Views")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("**Views vs Likes (Engagement)**")
        v = vids_df.dropna(subset=["likes"]).copy()
        if not v.empty:
            fig2, ax2 = plt.subplots(figsize=(6,6))
            ax2.scatter(v["views"], v["likes"], alpha=0.6)
            ax2.set_xlabel("Views")
            ax2.set_ylabel("Likes")
            ax2.set_title("Views vs Likes")
            st.pyplot(fig2)
        else:
            st.info("No like data available to plot.")

    st.markdown("**Video Duration Distribution (minutes)**")
    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.hist(vids_df["duration_min"].dropna(), bins=30)
    ax3.set_xlabel("Duration (minutes)")
    ax3.set_ylabel("Count")
    ax3.set_title("Video Duration Distribution")
    st.pyplot(fig3)

    st.markdown("**Upload Activity (Day of Week × Hour)**")
    tmp = vids_df.dropna(subset=["published_at"]).copy()
    if not tmp.empty:
        tmp["dow"] = tmp["published_at"].dt.day_name()
        tmp["hour"] = tmp["published_at"].dt.hour
        piv = tmp.pivot_table(index="dow", columns="hour", values="video_id", aggfunc="count").fillna(0)
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        piv = piv.reindex(day_order)
        fig4, ax4 = plt.subplots(figsize=(12,5))
        im = ax4.imshow(piv.values, aspect="auto")
        ax4.set_yticks(range(len(piv.index)))
        ax4.set_yticklabels(piv.index)
        ax4.set_xticks(range(len(piv.columns)))
        ax4.set_xticklabels(piv.columns)
        ax4.set_xlabel("Hour of Day (UTC)")
        ax4.set_ylabel("Day of Week")
        ax4.set_title("Upload Activity (counts)")
        plt.colorbar(im, ax=ax4, fraction=0.02, pad=0.04)
        st.pyplot(fig4)
    else:
        st.info("No publish time data to plot.")

    st.markdown("**Title Keyword WordCloud**")
    titles_text = " ".join([str(t) for t in vids_df["title"].dropna().tolist()])
    fig_wc = build_wordcloud(titles_text)
    if fig_wc is not None:
        st.pyplot(fig_wc)
    else:
        st.info("Not enough textual data for word cloud.")

    st.success("Analysis complete ✅")

else:
    st.info("👈 Enter channels in the sidebar and click **Fetch & Analyze** to begin.")
