"""
Trend Momentum Analysis Module
Computes trend momentum scores over time and generates Plotly visualizations
for Reddit and Twitter post data.
"""

import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go


def compute_momentum(df, platform_name="Platform"):
    """
    Compute a momentum score for each post.
    momentum_score = engagement / hours_since_post
    Higher score = more recent AND high engagement = strong momentum.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    now = time.time()

    # Hours since post was created
    df['hours_ago'] = (now - df['created_utc']) / 3600.0
    df['hours_ago'] = df['hours_ago'].clip(lower=0.1)  # avoid division by zero

    # Engagement metric
    if 'reposts' in df.columns:
        df['engagement'] = df['upvotes'] + df.get('comments', pd.Series(0)) + df['reposts']
    else:
        df['engagement'] = df['upvotes'] + df.get('comments', pd.Series(0))

    # Momentum score
    df['momentum'] = df['engagement'] / df['hours_ago']

    # Time bucket
    df['time_bucket'] = pd.cut(
        df['hours_ago'],
        bins=[0, 1, 6, 24, 72, float('inf')],
        labels=['Last 1h', 'Last 6h', 'Last 24h', 'Last 3 days', 'Older'],
        ordered=True
    )

    df['platform'] = platform_name
    return df


def get_momentum_line_chart(reddit_df, twitter_df):
    """
    Creates a line chart showing trend momentum over time for both platforms.
    """
    frames = []
    if reddit_df is not None and not reddit_df.empty:
        r = compute_momentum(reddit_df, "Reddit")
        if not r.empty:
            frames.append(r)
    if twitter_df is not None and not twitter_df.empty:
        t = compute_momentum(twitter_df, "Twitter")
        if not t.empty:
            frames.append(t)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('hours_ago')

    # Bucket-level aggregation for a clean line chart
    bucket_order = ['Last 1h', 'Last 6h', 'Last 24h', 'Last 3 days', 'Older']
    agg = combined.groupby(['time_bucket', 'platform'], observed=True)['momentum'].mean().reset_index()
    agg['time_bucket'] = pd.Categorical(agg['time_bucket'], categories=bucket_order, ordered=True)
    agg = agg.sort_values('time_bucket')

    fig = px.line(
        agg, x='time_bucket', y='momentum', color='platform',
        markers=True,
        labels={'time_bucket': 'Time Window', 'momentum': 'Avg Momentum Score', 'platform': 'Platform'},
        color_discrete_map={'Reddit': '#FF4500', 'Twitter': '#1DA1F2'}
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def get_platform_comparison_chart(reddit_df, twitter_df):
    """
    Bar chart comparing total engagement between Reddit and Twitter.
    """
    data = []

    if reddit_df is not None and not reddit_df.empty:
        total_upvotes = reddit_df['upvotes'].sum()
        total_comments = reddit_df['comments'].sum() if 'comments' in reddit_df.columns else 0
        data.append({'Platform': 'Reddit', 'Metric': 'Upvotes', 'Value': int(total_upvotes)})
        data.append({'Platform': 'Reddit', 'Metric': 'Comments', 'Value': int(total_comments)})

    if twitter_df is not None and not twitter_df.empty:
        total_likes = twitter_df['upvotes'].sum()
        total_reposts = twitter_df['reposts'].sum() if 'reposts' in twitter_df.columns else 0
        data.append({'Platform': 'Twitter', 'Metric': 'Likes', 'Value': int(total_likes)})
        data.append({'Platform': 'Twitter', 'Metric': 'Reposts', 'Value': int(total_reposts)})

    if not data:
        return None

    df = pd.DataFrame(data)
    fig = px.bar(
        df, x='Metric', y='Value', color='Platform', barmode='group',
        color_discrete_map={'Reddit': '#FF4500', 'Twitter': '#1DA1F2'}
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def get_engagement_distribution_chart(reddit_df, twitter_df):
    """
    Histogram showing the distribution of engagement scores across all posts.
    """
    frames = []
    if reddit_df is not None and not reddit_df.empty:
        r = reddit_df[['upvotes']].copy()
        r['Platform'] = 'Reddit'
        r.rename(columns={'upvotes': 'Engagement'}, inplace=True)
        frames.append(r)
    if twitter_df is not None and not twitter_df.empty:
        t = twitter_df[['upvotes']].copy()
        t['Platform'] = 'Twitter'
        t.rename(columns={'upvotes': 'Engagement'}, inplace=True)
        frames.append(t)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    fig = px.histogram(
        combined, x='Engagement', color='Platform', nbins=20,
        barmode='overlay', opacity=0.7,
        color_discrete_map={'Reddit': '#FF4500', 'Twitter': '#1DA1F2'},
        labels={'Engagement': 'Engagement Score (Upvotes/Likes)', 'count': 'Number of Posts'}
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig
