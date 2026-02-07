"""
Enhanced Cross-Platform Trend Synthesis Module (v2)

This module provides a higher-quality alternative to the baseline cross-platform
synthesis logic. It is designed to be:
- Keyword-grounded (explicitly mentions the input keyword)
- Concise (4-6 sentences total)
- Structured and readable
- Focused on recent activity and momentum
- Platform-aware (distinct insights per platform)

Author: Senior ML + NLP Engineer
Purpose: Academic demo and panel explanation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple


# ============================================================================
# SIGNAL EXTRACTION FUNCTIONS
# ============================================================================

def extract_twitter_signals(df: pd.DataFrame, keyword: str) -> Dict:
    """
    Extract Twitter/X-specific signals focusing on velocity, sentiment, and breaking news.
    
    Args:
        df: DataFrame with Twitter posts (columns: text, upvotes, comments, reposts, created_utc)
        keyword: The search keyword for context
        
    Returns:
        Dict with keys: velocity, sentiment_tone, top_insight, strength
    """
    signals = {
        "velocity": "low",
        "sentiment_tone": "neutral",
        "top_insight": None,
        "strength": 0.0
    }
    
    if df.empty:
        return signals
    
    # Calculate velocity based on recent activity (last 3 days vs older)
    now = datetime.now().timestamp()
    three_days_ago = now - (3 * 86400)
    
    recent_posts = df[df['created_utc'] >= three_days_ago]
    older_posts = df[df['created_utc'] < three_days_ago]
    
    recent_count = len(recent_posts)
    older_count = len(older_posts)
    
    # Velocity classification
    if recent_count > older_count * 1.5:
        signals["velocity"] = "surging"
    elif recent_count > older_count:
        signals["velocity"] = "growing"
    elif recent_count > 0:
        signals["velocity"] = "steady"
    
    # Engagement strength (normalized)
    if not df.empty:
        avg_engagement = df['upvotes'].mean() + df.get('reposts', pd.Series([0])).mean()
        max_engagement = df['upvotes'].max() + df.get('reposts', pd.Series([0])).max()
        
        # Strength score (0-1)
        signals["strength"] = min(1.0, avg_engagement / max(1, max_engagement * 0.3))
    
    # Sentiment analysis (basic heuristic based on engagement patterns)
    if not df.empty:
        # High repost-to-like ratio suggests viral/controversial content
        if 'reposts' in df.columns:
            repost_ratio = df['reposts'].sum() / max(1, df['upvotes'].sum())
            if repost_ratio > 0.25:
                signals["sentiment_tone"] = "viral"
            elif repost_ratio > 0.15:
                signals["sentiment_tone"] = "engaged"
    
    # Extract top insight from most engaged post
    if not df.empty:
        top_post = df.nlargest(1, 'upvotes').iloc[0]
        signals["top_insight"] = top_post['text']
    
    return signals


def extract_reddit_signals(df: pd.DataFrame, keyword: str) -> Dict:
    """
    Extract Reddit-specific signals focusing on depth, discussion quality, and context.
    
    Args:
        df: DataFrame with Reddit posts (columns: text, upvotes, comments, created_utc)
        keyword: The search keyword for context
        
    Returns:
        Dict with keys: discussion_depth, context_type, top_insight, strength
    """
    signals = {
        "discussion_depth": "low",
        "context_type": "general",
        "top_insight": None,
        "strength": 0.0
    }
    
    if df.empty:
        return signals
    
    # Discussion depth based on comment-to-upvote ratio
    if not df.empty and 'comments' in df.columns:
        avg_comment_ratio = df['comments'].sum() / max(1, df['upvotes'].sum())
        
        if avg_comment_ratio > 0.15:
            signals["discussion_depth"] = "high"
        elif avg_comment_ratio > 0.08:
            signals["discussion_depth"] = "moderate"
    
    # Context type detection (heuristic based on post titles)
    if not df.empty:
        combined_text = ' '.join(df['text'].head(10).tolist()).lower()
        
        # Check for analytical/explanatory keywords
        if any(word in combined_text for word in ['eli5', 'explain', 'why', 'how', 'analysis', 'comparison']):
            signals["context_type"] = "analytical"
        elif any(word in combined_text for word in ['unpopular opinion', 'debate', 'change my view', 'cmv']):
            signals["context_type"] = "debate"
        elif any(word in combined_text for word in ['til', 'did you know', 'fact', 'history']):
            signals["context_type"] = "educational"
    
    # Strength based on engagement
    if not df.empty:
        avg_upvotes = df['upvotes'].mean()
        max_upvotes = df['upvotes'].max()
        signals["strength"] = min(1.0, avg_upvotes / max(1, max_upvotes * 0.3))
    
    # Extract top insight from most discussed post
    if not df.empty:
        # Prioritize posts with high comment counts
        if 'comments' in df.columns and df['comments'].max() > 0:
            top_post = df.nlargest(1, 'comments').iloc[0]
        else:
            top_post = df.nlargest(1, 'upvotes').iloc[0]
        signals["top_insight"] = top_post['text']
    
    return signals


def extract_news_signals(df: Optional[pd.DataFrame], keyword: str) -> Dict:
    """
    Extract news-specific signals (placeholder for future news integration).
    
    Args:
        df: DataFrame with news articles (currently None)
        keyword: The search keyword for context
        
    Returns:
        Dict with keys: factual_grounding, impact, top_insight, strength
    """
    signals = {
        "factual_grounding": None,
        "impact": None,
        "top_insight": None,
        "strength": 0.0
    }
    
    # Placeholder for future news API integration
    # When implemented, this would extract:
    # - Real-world events/announcements
    # - Official statements
    # - Industry impact
    
    return signals


# ============================================================================
# MOMENTUM DETECTION
# ============================================================================

def detect_momentum(twitter_df: pd.DataFrame, reddit_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect overall momentum and recency signals across platforms.
    
    Args:
        twitter_df: Twitter posts DataFrame
        reddit_df: Reddit posts DataFrame
        
    Returns:
        Tuple of (momentum_level, momentum_phrase)
        - momentum_level: "high", "moderate", "low", "none"
        - momentum_phrase: Human-readable description for report
    """
    momentum_level = "none"
    momentum_phrase = "limited recent activity"
    
    now = datetime.now().timestamp()
    one_day_ago = now - 86400
    three_days_ago = now - (3 * 86400)
    week_ago = now - (7 * 86400)
    
    # Count recent posts across platforms
    recent_24h = 0
    recent_3d = 0
    recent_7d = 0
    
    for df in [twitter_df, reddit_df]:
        if not df.empty and 'created_utc' in df.columns:
            recent_24h += len(df[df['created_utc'] >= one_day_ago])
            recent_3d += len(df[df['created_utc'] >= three_days_ago])
            recent_7d += len(df[df['created_utc'] >= week_ago])
    
    # Classify momentum
    if recent_24h > 10:
        momentum_level = "high"
        momentum_phrase = "experiencing a significant surge in discussion over the past 24 hours"
    elif recent_3d > 15:
        momentum_level = "moderate"
        momentum_phrase = "showing renewed attention with growing discussion in recent days"
    elif recent_7d > 10:
        momentum_level = "low"
        momentum_phrase = "maintaining steady but modest engagement this week"
    else:
        momentum_level = "none"
        momentum_phrase = "showing limited recent activity across platforms"
    
    return momentum_level, momentum_phrase


# ============================================================================
# STRUCTURED REPORT GENERATOR
# ============================================================================

def generate_cross_platform_report(
    keyword: str,
    twitter_df: pd.DataFrame,
    reddit_df: pd.DataFrame,
    news_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate a concise, structured cross-platform trend report.
    
    FIXED STRUCTURE (4-6 sentences):
    - Lines 1-2: Keyword-grounded introduction + dominant theme
    - Line 3: Momentum/recency signal
    - Lines 4-5: Cross-platform synthesis (Twitter → Reddit → News)
    
    Args:
        keyword: The search keyword (MUST be mentioned in output)
        twitter_df: Twitter posts DataFrame
        reddit_df: Reddit posts DataFrame
        news_df: News articles DataFrame (optional, for future use)
        
    Returns:
        Structured report string (4-6 sentences, no bullet points)
    """
    
    # Extract platform-specific signals
    twitter_signals = extract_twitter_signals(twitter_df, keyword)
    reddit_signals = extract_reddit_signals(reddit_df, keyword)
    news_signals = extract_news_signals(news_df, keyword)
    
    # Detect momentum
    momentum_level, momentum_phrase = detect_momentum(twitter_df, reddit_df)
    
    # Determine dominant platform and theme
    twitter_strength = twitter_signals["strength"]
    reddit_strength = reddit_signals["strength"]
    
    # Build report components
    sentences = []
    
    # ========== LINES 1-2: Keyword-grounded introduction ==========
    if twitter_strength > reddit_strength and twitter_strength > 0.3:
        dominant_platform = "Twitter/X"
        theme = "rapid reactions and viral sharing"
    elif reddit_strength > 0.3:
        dominant_platform = "Reddit"
        theme = "in-depth discussions and community analysis"
    else:
        dominant_platform = "social media"
        theme = "emerging conversations"
    
    # Sentence 1: Keyword + dominant platform
    intro = f'The keyword "{keyword}" is currently generating {theme} primarily on {dominant_platform}.'
    sentences.append(intro)
    
    # Sentence 2: Context about the discussion
    if reddit_signals["context_type"] != "general":
        context = f'The discourse is predominantly {reddit_signals["context_type"]} in nature, with users seeking deeper understanding of the topic.'
    elif twitter_signals["sentiment_tone"] == "viral":
        context = f'The conversation has taken on a viral quality, with high sharing velocity indicating widespread interest.'
    else:
        context = f'The discussion spans multiple perspectives, reflecting diverse community engagement with the subject.'
    sentences.append(context)
    
    # ========== LINE 3: Momentum/recency signal ==========
    momentum_sentence = f'Activity patterns indicate the topic is {momentum_phrase}.'
    sentences.append(momentum_sentence)
    
    # ========== LINES 4-5: Cross-platform synthesis ==========
    
    # Twitter insight (if strong signal)
    if twitter_strength > 0.2:
        if twitter_signals["velocity"] == "surging":
            twitter_insight = f'On Twitter/X, the trend is surging with {twitter_signals["velocity"]} engagement, suggesting breaking developments or renewed public interest.'
        elif twitter_signals["velocity"] == "growing":
            twitter_insight = f'Twitter/X shows growing traction with {twitter_signals["sentiment_tone"]} sentiment, indicating building momentum.'
        else:
            twitter_insight = f'Twitter/X activity remains {twitter_signals["velocity"]}, with reactions primarily characterized by {twitter_signals["sentiment_tone"]} engagement.'
        sentences.append(twitter_insight)
    
    # Reddit insight (if strong signal)
    if reddit_strength > 0.2:
        if reddit_signals["discussion_depth"] == "high":
            reddit_insight = f'Reddit communities are contributing {reddit_signals["discussion_depth"]}-depth analysis, with {reddit_signals["context_type"]} threads providing contextual grounding and comparative perspectives.'
        elif reddit_signals["discussion_depth"] == "moderate":
            reddit_insight = f'Reddit discussions show {reddit_signals["discussion_depth"]} engagement, with users exploring {reddit_signals["context_type"]} dimensions of the topic.'
        else:
            reddit_insight = f'Reddit activity is present but limited, with early discussions suggesting {reddit_signals["context_type"]} interest.'
        sentences.append(reddit_insight)
    
    # Fallback if both platforms are weak
    if twitter_strength <= 0.2 and reddit_strength <= 0.2:
        weak_signal = f'Both platforms show relatively weak signals, suggesting the topic may be niche, emerging, or experiencing a lull in public attention.'
        sentences.append(weak_signal)
    
    # News insight (placeholder for future)
    if news_signals["strength"] > 0.2 and news_signals["top_insight"]:
        news_insight = f'News coverage provides factual grounding, with reports highlighting real-world implications and industry impact.'
        sentences.append(news_insight)
    
    # ========== FINAL ASSEMBLY ==========
    # Ensure 4-6 sentences (trim if needed)
    if len(sentences) > 6:
        sentences = sentences[:6]
    
    # Join with proper spacing
    report = ' '.join(sentences)
    
    return report


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = ['generate_cross_platform_report']
