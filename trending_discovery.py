"""
Trending Topics Discovery Module (Enhanced with Better Categorization)

This module discovers trending topics by leveraging Reddit's API and categorizes
them into user-friendly categories with improved relevance filtering.

Strategy:
- Fetch top posts from Reddit's r/all/hot for real-time trending topics
- Filter for relevant, newsworthy content (not memes or generic posts)
- Categorize posts into: Technology, Sports, Entertainment, Politics, News
- Extract specific, meaningful keywords (not generic phrases)
- Return top 10 with balanced category distribution
"""

import sys
import os
import re
from datetime import datetime, timedelta

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(__file__))

# Category definitions with improved keyword mappings
CATEGORY_KEYWORDS = {
    'Technology': {
        'keywords': ['ai', 'artificial intelligence', 'chatgpt', 'openai', 'tech', 'software', 
                    'app', 'computer', 'iphone', 'android', 'google', 'apple', 'microsoft', 
                    'meta', 'tesla', 'spacex', 'crypto', 'bitcoin', 'ethereum', 'blockchain', 
                    'nvidia', 'amd', 'intel', 'samsung', 'programming', 'developer', 'startup',
                    'cybersecurity', 'hacker', 'data breach', 'robot', 'automation', 'drone'],
        'subreddits': ['technology', 'gadgets', 'android', 'apple', 'cryptocurrency', 
                      'bitcoin', 'teslamotors', 'spacex', 'programming', 'coding']
    },
    'Sports': {
        'keywords': ['nba', 'nfl', 'soccer', 'football', 'basketball', 'baseball', 'hockey', 
                    'nhl', 'mlb', 'tennis', 'cricket', 'golf', 'boxing', 'ufc', 'mma', 
                    'olympics', 'fifa', 'world cup', 'super bowl', 'playoff', 'championship',
                    'athlete', 'player', 'coach', 'team', 'league', 'tournament', 'f1', 
                    'formula 1', 'racing', 'motorsport', 'skiing', 'snowboard', 'winter sports'],
        'subreddits': ['sports', 'nba', 'nfl', 'soccer', 'football', 'hockey', 'baseball',
                      'tennis', 'cricket', 'mma', 'boxing', 'formula1', 'olympics']
    },
    'Entertainment': {
        'keywords': ['movie', 'film', 'netflix', 'disney', 'marvel', 'star wars', 'hbo',
                    'tv show', 'series', 'season', 'episode', 'actor', 'actress', 'director',
                    'music', 'album', 'song', 'concert', 'tour', 'grammy', 'oscar', 'emmy',
                    'celebrity', 'hollywood', 'anime', 'manga', 'gaming', 'game', 'xbox', 
                    'playstation', 'nintendo', 'esports', 'streamer', 'youtube', 'tiktok'],
        'subreddits': ['movies', 'television', 'music', 'anime', 'marvel', 'starwars',
                      'entertainment', 'netflix', 'hbo', 'gaming', 'games', 'pcgaming']
    },
    'Politics': {
        'keywords': ['election', 'president', 'congress', 'senate', 'house', 'government', 
                    'political', 'democrat', 'republican', 'vote', 'voting', 'law', 'court',
                    'supreme court', 'policy', 'biden', 'trump', 'parliament', 'minister',
                    'campaign', 'bill', 'legislation', 'protest', 'rally', 'impeachment',
                    'scandal', 'investigation', 'hearing', 'testimony'],
        'subreddits': ['politics', 'worldpolitics', 'conservative', 'liberal', 'libertarian',
                      'politicaldiscussion']
    },
    'News': {
        'keywords': ['breaking', 'breaking news', 'report', 'announced', 'announces', 'update',
                    'alert', 'emergency', 'crisis', 'disaster', 'climate', 'climate change',
                    'weather', 'storm', 'hurricane', 'earthquake', 'wildfire', 'war', 'conflict',
                    'ukraine', 'russia', 'china', 'economy', 'recession', 'inflation', 'stock',
                    'market crash', 'unemployment', 'pandemic', 'covid', 'vaccine', 'outbreak',
                    'study', 'research', 'science', 'discovery', 'nasa', 'space'],
        'subreddits': ['news', 'worldnews', 'science', 'space', 'health', 'economics', 
                      'environment', 'climate']
    }
}

# Subreddits to exclude (memes, low-quality, NSFW, controversial)
EXCLUDED_SUBREDDITS = {
    # Memes and low-quality
    'memes', 'dankmemes', 'me_irl', 'meirl', 'funny', 'pics', 'aww', 'wholesomememes',
    'adviceanimals', 'terriblefacebookmemes', 'comedyheaven', 'okbuddyretard',
    'shitposting', 'copypasta', 'circlejerk', 'teenagers', 'askouija',
    # NSFW and adult content
    'nsfw', 'nsfw_gif', 'gonewild', 'realgirls', 'celebnsfw', 'nsfw411',
    # Controversial/sensitive
    'conspiracy', 'conspiracytheories', 'mensrights', 'theredpill', 'incels',
    'drama', 'subredditdrama', 'publicfreakout', 'actualpublicfreakouts',
    # Relationship/dating (can be awkward)
    'relationship_advice', 'amitheasshole', 'tifu', 'confessions',
    # Drugs and alcohol
    'trees', 'drugs', 'lsd', 'cocaine', 'opiates', 'drunk'
}

# Generic/vague words that make bad keywords
GENERIC_WORDS = {
    'plan', 'thing', 'stuff', 'something', 'someone', 'people', 'person', 'way',
    'time', 'day', 'year', 'today', 'yesterday', 'tomorrow', 'just', 'really',
    'very', 'much', 'many', 'some', 'all', 'every', 'never', 'always', 'core',
    'memory', 'fades', 'seeks', 'expedited', 'dhs'
}

# Profanity and inappropriate content filter
PROFANITY_KEYWORDS = {
    # Common profanity (partial list for demo safety)
    'fuck', 'shit', 'damn', 'hell', 'ass', 'bitch', 'bastard', 'crap',
    'piss', 'dick', 'cock', 'pussy', 'whore', 'slut', 'fag', 'retard',
    # Variations
    'wtf', 'stfu', 'nsfw', 'milf', 'porn', 'xxx', 'sex', 'sexy',
    # Potentially offensive
    'kill', 'murder', 'suicide', 'rape', 'nazi', 'hitler',
    # Drug references
    'weed', 'marijuana', 'cocaine', 'meth', 'heroin', 'drug dealer',
    # Typos/misspellings that slip through
    'pubic'  # Common typo for "public" but inappropriate
}

# Controversial topics to avoid in professional demos
CONTROVERSIAL_KEYWORDS = {
    'abortion', 'gun control', 'immigration ban', 'racist', 'sexist',
    'transgender bathroom', 'antifa', 'proud boys', 'qanon',
    'vaccine mandate', 'anti-vax', 'flat earth'
}

# Political figures and controversial names (avoid in demos)
POLITICAL_FIGURES = {
    'trump', 'donald trump', 'biden', 'joe biden', 'obama', 'clinton',
    'hillary', 'bernie', 'sanders', 'desantis', 'pence', 'kamala',
    'epstein', 'jeffrey epstein', 'ghislaine', 'weinstein', 'cosby',
    'vance', 'jd vance', 'pelosi', 'mcconnell', 'aoc', 'ocasio-cortez'
}

# Political party and government terms (avoid political drama)
POLITICAL_TERMS = {
    'republicans', 'democrats', 'gop', 'dnc', 'rnc',
    'congress', 'senate', 'house of representatives',
    'impeachment', 'indictment', 'investigation',
    'scandal', 'corruption', 'bribery'
}


def is_professional_content(title, subreddit):
    """
    Filter out profanity, NSFW, and controversial content for professional demos.
    Enhanced to catch masked profanity and political content.
    
    Args:
        title: Post title
        subreddit: Subreddit name
        
    Returns:
        Boolean - True if content is professional/demo-safe
    """
    title_lower = title.lower()
    subreddit_lower = subreddit.lower()
    
    # Check for asterisk-masked profanity (f***, s***, etc.)
    masked_profanity_patterns = [
        r'\bf\*+\w*\b',  # f***, f**k, etc.
        r'\bs\*+\w*\b',  # s***, s**t, etc.
        r'\ba\*+s\b',    # a**
        r'\bb\*+ch\b',   # b***h
        r'\bd\*+k\b',    # d**k
    ]
    for pattern in masked_profanity_patterns:
        if re.search(pattern, title_lower):
            return False
    
    # Check for profanity in title (original check)
    for profanity in PROFANITY_KEYWORDS:
        # Use word boundaries to avoid false positives
        pattern = r'\b' + re.escape(profanity) + r'\w*\b'
        if re.search(pattern, title_lower):
            return False
    
    # Check for political figures (avoid political drama)
    for figure in POLITICAL_FIGURES:
        if figure in title_lower:
            return False
    
    # Check for political party terms and government drama
    for term in POLITICAL_TERMS:
        if term in title_lower:
            return False
    
    # Check for controversial topics
    for controversial in CONTROVERSIAL_KEYWORDS:
        if controversial in title_lower:
            return False
    
    # Additional NSFW indicators in title
    nsfw_indicators = ['nsfw', '[nsfw]', '(nsfw)', 'not safe for work', 'explicit']
    for indicator in nsfw_indicators:
        if indicator in title_lower:
            return False
    
    return True



def is_relevant_post(title, subreddit, engagement_score):
    """
    Filter out irrelevant, meme, or low-quality posts.
    
    Args:
        title: Post title
        subreddit: Subreddit name
        engagement_score: Engagement score
        
    Returns:
        Boolean - True if post is relevant
    """
    title_lower = title.lower()
    subreddit_lower = subreddit.lower()
    
    # Exclude meme/joke/NSFW subreddits
    if subreddit_lower in EXCLUDED_SUBREDDITS:
        return False
    
    # Professional content filter (profanity, NSFW, controversial)
    if not is_professional_content(title, subreddit):
        return False
    
    # Require minimum engagement for relevance
    if engagement_score < 500:
        return False
    
    # Filter out posts that are too short (likely low-quality)
    if len(title.split()) < 4:
        return False
    
    # Filter out posts with excessive punctuation (clickbait)
    if title.count('!') > 2 or title.count('?') > 2:
        return False
    
    # Filter out posts that are all caps (spam/clickbait)
    if title.isupper() and len(title) > 10:
        return False
    
    return True


def categorize_post(title, subreddit):
    """
    Categorize a post based on title keywords and subreddit with improved accuracy.
    
    Args:
        title: Post title (string)
        subreddit: Subreddit name (string)
        
    Returns:
        Category name (string) or 'General' if no match
    """
    title_lower = title.lower()
    subreddit_lower = subreddit.lower()
    
    # Score each category
    category_scores = {}
    
    for category, data in CATEGORY_KEYWORDS.items():
        score = 0
        
        # Check subreddit match (very strong signal - weight heavily)
        if subreddit_lower in data['subreddits']:
            score += 20  # Increased from 10
        
        # Check keyword matches in title (with word boundaries for accuracy)
        for keyword in data['keywords']:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, title_lower):
                score += 2  # Increased from 1
        
        category_scores[category] = score
    
    # Return category with highest score, or 'General' if score is too low
    max_score = max(category_scores.values())
    if max_score >= 3:  # Require minimum score for categorization
        return max(category_scores, key=category_scores.get)
    else:
        return 'General'


def discover_trending_topics(limit=10):
    """
    Discover trending topics with improved relevance and categorization.
    
    Args:
        limit: Number of trending topics to return (default: 10)
        
    Returns:
        List of dicts with trending topic data
    """
    import requests
    from collections import defaultdict
    
    all_posts = []
    
    try:
        # Fetch trending posts from Reddit's r/all/hot
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        url = "https://www.reddit.com/r/all/hot.json?limit=100"
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            posts = data.get("data", {}).get("children", [])
            
            # Extract and filter posts
            for post in posts:
                p = post.get("data", {})
                
                # Calculate engagement score
                upvotes = p.get("score", 0)
                comments = p.get("num_comments", 0)
                engagement_score = upvotes + (comments * 2)
                
                # Extract title and subreddit
                title = p.get("title", "")
                subreddit = p.get("subreddit", "unknown")
                
                # Skip NSFW
                if p.get("over_18", False):
                    continue
                
                # Filter for relevance
                if not is_relevant_post(title, subreddit, engagement_score):
                    continue
                
                # Categorize the post
                category = categorize_post(title, subreddit)
                
                # Extract keyword from title
                keyword = extract_keyword_from_title(title)
                
                # Skip if keyword is too generic
                if keyword.lower() in GENERIC_WORDS or len(keyword) < 3:
                    continue
                
                all_posts.append({
                    'keyword': keyword,
                    'title': title[:80] + "..." if len(title) > 80 else title,
                    'category': category,
                    'platform': 'Reddit',
                    'engagement_score': engagement_score,
                    'url': f"https://reddit.com{p.get('permalink', '')}",
                    'subreddit': subreddit
                })
        
        # Sort all posts by engagement
        all_posts.sort(key=lambda x: x['engagement_score'], reverse=True)
        
        # Select diverse topics across categories
        trending_items = select_diverse_topics(all_posts, limit)
        
        # Add rank
        for idx, item in enumerate(trending_items, 1):
            item['rank'] = idx
            
    except Exception as e:
        print(f"Error fetching trending topics: {e}")
        trending_items = []
    
    return trending_items


def select_diverse_topics(posts, limit=10):
    """
    Select diverse topics ensuring representation across categories.
    """
    from collections import defaultdict
    
    selected = []
    category_counts = defaultdict(int)
    
    # Define max posts per category (max 30% from one category)
    max_per_category = max(2, limit // 3)
    
    # First pass: Select top posts while maintaining diversity
    for post in posts:
        category = post['category']
        
        # Skip if we've hit the category limit
        if category_counts[category] >= max_per_category:
            continue
        
        selected.append(post)
        category_counts[category] += 1
        
        if len(selected) >= limit:
            break
    
    # Second pass: Fill remaining slots if needed
    if len(selected) < limit:
        for post in posts:
            if post not in selected:
                selected.append(post)
                if len(selected) >= limit:
                    break
    
    return selected[:limit]


def extract_keyword_from_title(title):
    """
    Extract a specific, meaningful keyword from a post title.
    Improved to get actual topic names, not generic phrases.
    """
    # Remove common Reddit prefixes
    title = re.sub(r'^(TIL|ELI5|TIFU|LPT|PSA|AMA|IAMA):\s*', '', title, flags=re.IGNORECASE)
    
    # Look for quoted text first (often the main topic)
    quoted = re.findall(r'"([^"]+)"', title)
    if quoted and len(quoted[0].split()) <= 4:
        return quoted[0].title()
    
    # Look for proper nouns (capitalized words) - often the main subject
    words = title.split()
    proper_nouns = []
    for i, word in enumerate(words):
        # Clean word
        clean_word = re.sub(r'[^\w\s]', '', word)
        # Check if it's a proper noun (capitalized, not at start, not all caps)
        if (clean_word and clean_word[0].isupper() and 
            not clean_word.isupper() and 
            i > 0 and  # Not first word
            len(clean_word) > 2):
            proper_nouns.append(clean_word)
    
    # If we found 1-3 proper nouns, use them
    if 1 <= len(proper_nouns) <= 3:
        return ' '.join(proper_nouns)
    
    # Fallback: extract meaningful words
    title_clean = re.sub(r'[^\w\s]', ' ', title)
    title_clean = ' '.join(title_clean.split())
    
    # Stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my',
        'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'us', 'them',
        'just', 'now', 'new', 'first', 'last', 'next', 'after', 'before'
    }
    
    # Split and filter
    words = title_clean.lower().split()
    meaningful_words = [w for w in words if w not in stop_words and w not in GENERIC_WORDS and len(w) > 2]
    
    # Return first 1-3 meaningful words
    if len(meaningful_words) >= 3:
        keyword = ' '.join(meaningful_words[:3])
    elif len(meaningful_words) >= 2:
        keyword = ' '.join(meaningful_words[:2])
    elif len(meaningful_words) >= 1:
        keyword = meaningful_words[0]
    else:
        # Last resort: use first non-stop word from original title
        for word in title.split():
            clean = re.sub(r'[^\w]', '', word)
            if clean.lower() not in stop_words and len(clean) > 2:
                return clean.title()
        return "Trending"
    
    return keyword.title()


# Test function
if __name__ == "__main__":
    print("Testing improved trending discovery...\n")
    topics = discover_trending_topics(10)
    
    if topics:
        print(f"Found {len(topics)} trending topics:\n")
        for topic in topics:
            print(f"#{topic['rank']} [{topic['category']}] - {topic['keyword']}")
            print(f"   Title: {topic['title']}")
            print(f"   Subreddit: r/{topic['subreddit']}")
            print(f"   Engagement: {topic['engagement_score']:,}\n")
    else:
        print("No trending topics found.")
