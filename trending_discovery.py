"""
Trending Topics Discovery Module

This module discovers trending topics by leveraging existing Reddit and Twitter
data fetching infrastructure from backend_logic.py. It uses a curated list of
trending keywords and aggregates top posts by engagement.

Strategy:
- Fetch top posts from Reddit's r/all/hot for real-time trending topics
- Extract keywords from top posts
- Aggregate and rank by engagement (upvotes + comments)
- Return top 10 for sidebar display
"""

import sys
import os

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(__file__))

def discover_trending_topics(limit=10):
    """
    Discover trending topics using existing Reddit data fetching.
    
    This function fetches trending posts from Reddit's r/all and extracts
    the most engaging topics based on upvotes and comments.
    
    Args:
        limit: Number of trending topics to return (default: 10)
        
    Returns:
        List of dicts with structure:
        [
            {
                'rank': 1,
                'keyword': 'extracted keyword',
                'title': 'Post title',
                'platform': 'Reddit',
                'engagement_score': 45000,
                'url': 'https://...'
            },
            ...
        ]
    """
    import requests
    import re
    from collections import Counter
    
    trending_items = []
    
    try:
        # Fetch trending posts from Reddit's r/all/hot
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        url = "https://www.reddit.com/r/all/hot.json?limit=50"
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            posts = data.get("data", {}).get("children", [])
            
            # Extract and rank posts by engagement
            for post in posts:
                p = post.get("data", {})
                
                # Calculate engagement score
                upvotes = p.get("score", 0)
                comments = p.get("num_comments", 0)
                engagement_score = upvotes + (comments * 2)  # Weight comments higher
                
                # Extract title and clean it
                title = p.get("title", "")
                
                # Skip NSFW or low-quality posts
                if p.get("over_18", False) or engagement_score < 100:
                    continue
                
                # Extract keyword from title (first meaningful word or phrase)
                keyword = extract_keyword_from_title(title)
                
                trending_items.append({
                    'keyword': keyword,
                    'title': title[:80] + "..." if len(title) > 80 else title,
                    'platform': 'Reddit',
                    'engagement_score': engagement_score,
                    'url': f"https://reddit.com{p.get('permalink', '')}",
                    'subreddit': p.get('subreddit', 'unknown')
                })
        
        # Sort by engagement and take top items
        trending_items.sort(key=lambda x: x['engagement_score'], reverse=True)
        trending_items = trending_items[:limit]
        
        # Add rank
        for idx, item in enumerate(trending_items, 1):
            item['rank'] = idx
            
    except Exception as e:
        print(f"Error fetching trending topics: {e}")
        # Return empty list on error
        trending_items = []
    
    return trending_items


def extract_keyword_from_title(title):
    """
    Extract a meaningful keyword or phrase from a post title.
    
    Args:
        title: Post title string
        
    Returns:
        Extracted keyword (string)
    """
    import re
    
    # Remove common Reddit prefixes
    title = re.sub(r'^(TIL|ELI5|TIFU|LPT|PSA|AMA|IAMA):\s*', '', title, flags=re.IGNORECASE)
    
    # Remove special characters and extra spaces
    title = re.sub(r'[^\w\s]', ' ', title)
    title = ' '.join(title.split())
    
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my',
        'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'us', 'them'
    }
    
    # Split into words and filter
    words = title.lower().split()
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Return first 1-3 meaningful words as keyword
    if len(meaningful_words) >= 3:
        keyword = ' '.join(meaningful_words[:3])
    elif len(meaningful_words) >= 2:
        keyword = ' '.join(meaningful_words[:2])
    elif len(meaningful_words) >= 1:
        keyword = meaningful_words[0]
    else:
        # Fallback to first word of original title
        keyword = title.split()[0] if title.split() else "trending"
    
    return keyword.title()


# Test function for standalone execution
if __name__ == "__main__":
    print("Testing trending discovery...")
    topics = discover_trending_topics(10)
    
    if topics:
        print(f"\nFound {len(topics)} trending topics:\n")
        for topic in topics:
            print(f"#{topic['rank']} - {topic['keyword']}")
            print(f"   Title: {topic['title']}")
            print(f"   Engagement: {topic['engagement_score']:,}")
            print(f"   URL: {topic['url']}\n")
    else:
        print("No trending topics found.")
