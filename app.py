import streamlit as st
import time
import backend_logic as backend
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TrendScope AI",
    page_icon="TS",
    layout="wide",
    initial_sidebar_state="expanded"  # Show trending topics by default
)

# --- SESSION STATE FOR THEME ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# --- SESSION STATE FOR TRENDING TOPICS ---
if 'selected_trending' not in st.session_state:
    st.session_state.selected_trending = None
if 'auto_analyze' not in st.session_state:
    st.session_state.auto_analyze = False

def toggle_theme():
    if st.session_state.theme == 'dark':
        st.session_state.theme = 'light'
    else:
        st.session_state.theme = 'dark'

# --- CUSTOM CSS (ANIMATIONS & STYLING) ---
def inject_css():
    # Dynamic colors based on theme
    if st.session_state.theme == 'dark':
        bg_color = "#0e1117"
        text_color = "#fafafa"
        card_bg = "#1e2127"
        card_border = "#303339"
        accent_gradient = "linear-gradient(45deg, #FF4B4B, #FF914D)"
    else:
        bg_color = "#ffffff"  # Pure white background
        text_color = "#000000"
        card_bg = "#ffffff"   # White cards
        card_border = "#e0e0e0" # Light gray border for visibility
        accent_gradient = "linear-gradient(45deg, #4B8BFF, #4DFF91)"

    css = f"""
    <style>
        /* GLOBAL ANIMATIONS */
        @keyframes fade-in {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        
        /* HEADER STYLES */
        .main-header {{
            font-size: 3rem;
            font-weight: 800;
            background: {accent_gradient};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
            animation: fade-in 1s ease-out;
        }}
        
        /* CLEAN SEARCH INPUT */
        .stTextInput > div > div {{
            background-color: {card_bg};
            color: {text_color};
            border-radius: 12px;
            border: 1px solid {card_border};
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .stTextInput > div > div > input {{
            color: {text_color};
        }}
        
        /* CARD STYLES */
        .metric-card {{
            background-color: {card_bg};
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid {card_border};
            margin-bottom: 1rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            animation: fade-in 0.5s ease-out;
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
        }}
        
        /* LINK STYLING */
        a {{
            text-decoration: none;
            color: inherit;
        }}
        a:hover {{
            color: #FF4B4B;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css()

# --- SIDEBAR: TRENDING TOPICS ---
with st.sidebar:
    st.markdown("### 🔥 Trending Now")
    st.markdown("<small>Click any topic to analyze</small>", unsafe_allow_html=True)
    
    # Load trending topics (cached for performance)
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def load_trending():
        return backend.get_trending_topics(limit=10)
    
    trending_topics = load_trending()
    
    if trending_topics:
        for topic in trending_topics:
            # Create clickable button for each trending topic
            if st.button(
                f"#{topic['rank']} {topic['keyword']}",
                key=f"trending_{topic['rank']}",
                help=f"{topic['title'][:60]}...",
                use_container_width=True
            ):
                # Set selected trending topic and trigger analysis
                st.session_state.selected_trending = topic['keyword']
                st.session_state.auto_analyze = True
                st.rerun()
            
            # Show engagement score below button
            st.markdown(
                f"<small style='color: #888;'>💬 {topic['engagement_score']:,} engagement</small>",
                unsafe_allow_html=True
            )
            st.markdown("---")
    else:
        st.info("Loading trending topics...")
    
    # Add refresh button
    if st.button("🔄 Refresh Trending", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# --- HEADER & CONTROLS ---
col1, col2, col3 = st.columns([1, 6, 1])
with col3:
    # Theme Toggle Button
    btn_label = "Light" if st.session_state.theme == 'dark' else "Dark"
    if st.button(btn_label):
        toggle_theme()
        st.rerun()

st.markdown('<div class="main-header">TrendScope AI</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Unified Viral Trend Analysis & Prediction</p>", unsafe_allow_html=True)

# --- SEARCH INTERFACE ---
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    # Auto-populate from trending if selected
    default_keyword = st.session_state.selected_trending if st.session_state.selected_trending else ""
    keyword = st.text_input(
        "", 
        value=default_keyword,
        placeholder="Enter a trend, keyword, or hashtag...", 
        help="Type something and press Enter or click a trending topic"
    )
    
    # Enhanced synthesis toggle
    use_enhanced = st.checkbox(
        "Use Enhanced Cross-Platform Synthesis (v2)",
        value=False,
        help="Experimental: Keyword-grounded, momentum-focused synthesis with structured output"
    )
    
    search_clicked = st.button("Analyze Trend", use_container_width=True, type="primary")
    
    # Auto-trigger analysis if trending topic was clicked
    if st.session_state.auto_analyze:
        search_clicked = True
        st.session_state.auto_analyze = False  # Reset flag

# --- MAIN LOGIC ---
if keyword and search_clicked:
    # Progress Bar Animation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Scouting Reddit data...")
    progress_bar.progress(25)
    time.sleep(0.5) # UX delay
    
    status_text.text("Scouting X (Twitter) data...")
    progress_bar.progress(50)
    time.sleep(0.5)
    
    status_text.text("Processing Trend activity to obtain insights...")
    progress_bar.progress(75)
    
    # FETCH DATA
    try:
        data = backend.get_trend_data(keyword, use_enhanced_synthesis=use_enhanced)
        
        progress_bar.progress(100)
        status_text.empty()
        time.sleep(0.2)
        progress_bar.empty()
        
        # --- RESULTS DISPLAY ---
        
        st.markdown("---")
        
        # COLUMNS
        left_col, right_col = st.columns(2)
        
        # --- REDDIT COLUMN ---
        with left_col:
            st.markdown(f"### Reddit Analysis")
            r_data = data['reddit']
            
            # Metrics
            m1, m2 = st.columns(2)
            m1.metric("Viral Days", f"{r_data['viral_days']} days")
            m2.metric("Post Count", len(r_data['df']))
            
            # Summary Card
            st.markdown(f"""
            <div class="metric-card">
                <h4>AI Summary</h4>
                <p>{r_data['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Top Posts
            st.markdown("#### Top Discussions")
            if not r_data['df'].empty:
                for _, row in r_data['df'].sort_values("upvotes", ascending=False).head(5).iterrows():
                    # Truncate text for cleaner UI
                    display_text = row['text']
                    if len(display_text) > 60:
                        display_text = display_text[:60] + "..."
                        
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 10px;">
                        <b><a href="{row['url']}" target="_blank">{display_text}</a></b><br>
                        <small>Upvotes: {row['upvotes']} | Comments: {row['comments']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No Reddit data found.")

        # --- TWITTER COLUMN ---
        with right_col:
            st.markdown(f"### X (Twitter) Analysis")
            t_data = data['twitter']
            
            # Metrics
            m1, m2 = st.columns(2)
            m1.metric("Viral Days", f"{t_data['viral_days']} days")
            m2.metric("Post Count", len(t_data['df']))
            
            # Summary Card
            st.markdown(f"""
            <div class="metric-card">
                <h4>AI Summary</h4>
                <p>{t_data['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Top Posts
            st.markdown("#### Top Tweets")
            if not t_data['df'].empty:
                for _, row in t_data['df'].sort_values("upvotes", ascending=False).head(5).iterrows():
                    # Truncate text for cleaner UI
                    display_text = row['text']
                    if len(display_text) > 60:
                        display_text = display_text[:60] + "..."

                    st.markdown(f"""
                    <div class="metric-card" style="padding: 10px;">
                        <b><a href="{row['url']}" target="_blank">{display_text}</a></b><br>
                        <small>Likes: {row['upvotes']} | Reposts: {row['reposts']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No Twitter data found.")

        # --- CROSS-PLATFORM SUMMARY ---
        st.markdown("---")
        st.markdown("### Cross-Platform Synthesis")
        
        # Determine which summary to display
        summary_to_show = data.get('overall_summary_v2') if use_enhanced and data.get('overall_summary_v2') else data['overall_summary']
        version_label = " (Enhanced v2)" if use_enhanced and data.get('overall_summary_v2') else ""
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(90deg, rgba(255, 75, 75, 0.1) 0%, rgba(75, 139, 255, 0.1) 100%); border: 1px solid rgba(255,255,255,0.1);">
            <h3>Key Insight{version_label}</h3>
            <p style="font-size: 1.1rem;">{summary_to_show}</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
