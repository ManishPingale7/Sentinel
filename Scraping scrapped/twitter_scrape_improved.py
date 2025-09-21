# twitter_scrape_improved.py - Enhanced version with better bot detection avoidance
import snscrape.modules.twitter as sntwitter
import json
import time
import logging
import requests
from datetime import datetime
import random
import urllib3

# Disable SSL warnings (sometimes needed for snscrape)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twitter_scrape.log'),
        logging.StreamHandler()
    ]
)

# Configuration
QUERY = "flood OR flooded OR #flooding near:India"
OUTFILE = "tweets_stream.jsonl"

# Configure session with better headers
def setup_session():
    """Setup session with browser-like headers"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    return session

def stream_once_with_session(query, max_results=50, max_retries=3):
    """
    Enhanced scraping with session management and better error handling
    """
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries} - Setting up session...")
            
            # Set up custom session (this may help with some detection)
            session = setup_session()
            
            with open(OUTFILE, "a", encoding="utf-8") as f:
                logging.info(f"Starting scraping for query: {query}")
                
                tweets_collected = 0
                scraper = sntwitter.TwitterSearchScraper(query)
                
                # Add random delay before starting
                time.sleep(random.uniform(1, 5))
                
                for i, tweet in enumerate(scraper.get_items()):
                    try:
                        data = {
                            "id": tweet.id,
                            "datetime": tweet.date.isoformat(),
                            "user": tweet.user.username,
                            "content": tweet.content,
                            "place": tweet.place.name if tweet.place else None,
                            "lang": tweet.lang,
                            "scraped_at": datetime.now().isoformat()
                        }
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        tweets_collected += 1
                        
                        if i >= max_results - 1:
                            break
                        
                        # Random delay between tweets (human-like behavior)
                        time.sleep(random.uniform(0.5, 2.0))
                        
                    except Exception as tweet_error:
                        logging.warning(f"Error processing individual tweet: {tweet_error}")
                        continue
                
                logging.info(f"Successfully collected {tweets_collected} tweets")
                return tweets_collected
                
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Attempt {attempt + 1} failed: {error_msg}")
            
            # Check if it's a rate limit or access error
            if "4 requests" in error_msg or "ScraperException" in error_msg:
                logging.warning("Detected rate limiting or access restriction")
                
            if attempt < max_retries - 1:
                # Exponential backoff with more randomness
                base_wait = (2 ** attempt) * 120  # Start with 2 minutes, then 4, then 8
                jitter = random.uniform(0, 60)
                wait_time = base_wait + jitter
                logging.info(f"Waiting {wait_time/60:.1f} minutes before retry...")
                time.sleep(wait_time)
            else:
                logging.error("All retry attempts failed")
                return 0
    
    return 0

def main():
    logging.info("Starting Enhanced Twitter monitoring script...")
    logging.info(f"Query: {QUERY}")
    logging.info(f"Output file: {OUTFILE}")
    
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while True:
        try:
            start_time = datetime.now()
            tweets_collected = stream_once_with_session(QUERY, max_results=25)  # Very small batches
            
            if tweets_collected > 0:
                consecutive_failures = 0
                logging.info(f"Cycle completed successfully. Collected {tweets_collected} tweets.")
                # Longer sleep time to avoid rate limits
                sleep_time = random.uniform(900, 1200)  # 15-20 minutes
            else:
                consecutive_failures += 1
                logging.warning(f"No tweets collected. Consecutive failures: {consecutive_failures}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logging.error(f"Too many consecutive failures. Entering extended cooldown.")
                    sleep_time = random.uniform(3600, 7200)  # 1-2 hours
                    consecutive_failures = 0
                else:
                    sleep_time = random.uniform(1800, 2400)  # 30-40 minutes
            
            logging.info(f"Waiting {sleep_time/60:.1f} minutes before next cycle...")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logging.info("Script interrupted by user. Exiting gracefully...")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {str(e)}")
            time.sleep(random.uniform(1800, 3600))  # 30-60 minute cooldown

if __name__ == "__main__":
    main()