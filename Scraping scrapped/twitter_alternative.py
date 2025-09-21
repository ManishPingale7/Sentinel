# twitter_alternative.py - Alternative approaches for Twitter data collection
import json
import time
import logging
import requests
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twitter_alternative.log'),
        logging.StreamHandler()
    ]
)

class TwitterDataCollector:
    def __init__(self):
        self.outfile = "tweets_stream.jsonl"
        self.query_keywords = ["flood", "flooded", "flooding", "India"]
        
    def collect_with_bearer_token(self, bearer_token, max_results=100):
        """
        Method 1: Use Twitter API v2 with Bearer Token
        You need to get a bearer token from Twitter Developer Portal
        """
        url = "https://api.twitter.com/2/tweets/search/recent"
        
        query = "flood OR flooded OR flooding place:India -is:retweet"
        
        params = {
            'query': query,
            'max_results': min(max_results, 100),  # API limit
            'tweet.fields': 'created_at,author_id,public_metrics,lang,geo',
            'user.fields': 'username,location',
            'place.fields': 'name,country',
            'expansions': 'author_id,geo.place_id'
        }
        
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'User-Agent': 'TwitterDataCollector/1.0'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return self._process_api_response(data)
            else:
                logging.error(f"API request failed: {response.status_code} - {response.text}")
                return 0
                
        except Exception as e:
            logging.error(f"Error in API request: {str(e)}")
            return 0
    
    def collect_with_news_apis(self):
        """
        Method 2: Use news APIs as alternative data source
        Collect flood-related news from multiple sources
        """
        sources = [
            {
                'name': 'NewsAPI',
                'url': 'https://newsapi.org/v2/everything',
                'requires_key': True
            }
        ]
        
        # This is a placeholder - you'd need API keys for these services
        logging.info("Alternative: Collecting from news sources...")
        
        # Example with a free news API (replace with actual implementation)
        return self._collect_from_news_sources()
    
    def collect_with_rss_feeds(self):
        """
        Method 3: Collect from RSS feeds and news sources
        """
        rss_feeds = [
            'https://feeds.bbci.co.uk/news/world/asia/india/rss.xml',
            'https://timesofindia.indiatimes.com/rssfeeds/296589292.cms',
            'https://www.thehindu.com/news/national/?service=rss'
        ]
        
        collected = 0
        for feed_url in rss_feeds:
            try:
                collected += self._process_rss_feed(feed_url)
            except Exception as e:
                logging.error(f"Error processing RSS feed {feed_url}: {str(e)}")
        
        return collected
    
    def _process_api_response(self, data):
        """Process Twitter API v2 response"""
        tweets_saved = 0
        
        if 'data' not in data:
            logging.warning("No tweets found in API response")
            return 0
        
        # Create lookup dictionaries for users and places
        users = {}
        places = {}
        
        if 'includes' in data:
            if 'users' in data['includes']:
                users = {user['id']: user for user in data['includes']['users']}
            if 'places' in data['includes']:
                places = {place['id']: place for place in data['includes']['places']}
        
        with open(self.outfile, "a", encoding="utf-8") as f:
            for tweet in data['data']:
                # Get user info
                user_info = users.get(tweet['author_id'], {})
                
                # Get place info
                place_info = None
                if 'geo' in tweet and 'place_id' in tweet['geo']:
                    place_info = places.get(tweet['geo']['place_id'], {})
                
                processed_tweet = {
                    "id": tweet['id'],
                    "datetime": tweet['created_at'],
                    "user": user_info.get('username', 'unknown'),
                    "content": tweet['text'],
                    "place": place_info.get('name') if place_info else None,
                    "lang": tweet.get('lang', 'unknown'),
                    "source": "twitter_api_v2",
                    "scraped_at": datetime.now().isoformat()
                }
                
                f.write(json.dumps(processed_tweet, ensure_ascii=False) + "\n")
                tweets_saved += 1
        
        return tweets_saved
    
    def _collect_from_news_sources(self):
        """Collect flood-related news as alternative to tweets"""
        # This is a simplified example - you'd implement actual news collection here
        logging.info("Collecting from news sources (placeholder)")
        return 0
    
    def _process_rss_feed(self, feed_url):
        """Process RSS feed for flood-related content"""
        try:
            import feedparser
            
            feed = feedparser.parse(feed_url)
            flood_articles = 0
            
            with open(self.outfile, "a", encoding="utf-8") as f:
                for entry in feed.entries[:10]:  # Limit to recent entries
                    # Check if article is flood-related
                    content = f"{entry.title} {entry.summary}".lower()
                    if any(keyword in content for keyword in self.query_keywords):
                        article_data = {
                            "id": entry.id if hasattr(entry, 'id') else entry.link,
                            "datetime": entry.published if hasattr(entry, 'published') else datetime.now().isoformat(),
                            "user": "news_source",
                            "content": f"{entry.title}: {entry.summary}",
                            "place": "India",
                            "lang": "en",
                            "source": f"rss_{feed_url}",
                            "scraped_at": datetime.now().isoformat()
                        }
                        
                        f.write(json.dumps(article_data, ensure_ascii=False) + "\n")
                        flood_articles += 1
            
            return flood_articles
            
        except ImportError:
            logging.error("feedparser not installed. Install with: pip install feedparser")
            return 0
        except Exception as e:
            logging.error(f"RSS processing error: {str(e)}")
            return 0

def main():
    collector = TwitterDataCollector()
    
    # Check for Twitter API credentials
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    
    if bearer_token:
        logging.info("Found Twitter Bearer Token - using API v2")
        method = "api"
    else:
        logging.info("No Twitter credentials found - using alternative sources")
        method = "alternative"
    
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        try:
            start_time = datetime.now()
            collected = 0
            
            if method == "api" and bearer_token:
                collected = collector.collect_with_bearer_token(bearer_token, max_results=50)
            else:
                # Try RSS feeds as alternative
                collected = collector.collect_with_rss_feeds()
            
            if collected > 0:
                consecutive_failures = 0
                logging.info(f"Successfully collected {collected} items")
                sleep_time = 900  # 15 minutes
            else:
                consecutive_failures += 1
                logging.warning(f"No data collected. Consecutive failures: {consecutive_failures}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logging.error("Too many failures. Extended cooldown.")
                    sleep_time = 3600  # 1 hour
                    consecutive_failures = 0
                else:
                    sleep_time = 1800  # 30 minutes
            
            logging.info(f"Waiting {sleep_time/60:.1f} minutes before next cycle...")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logging.info("Script interrupted by user. Exiting...")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            time.sleep(1800)

if __name__ == "__main__":
    main()