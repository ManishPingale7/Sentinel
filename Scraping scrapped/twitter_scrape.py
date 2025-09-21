# twitter_snscrape_stream.py
import snscrape.modules.twitter as sntwitter
import json
import time
import logging
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twitter_scrape.log'),
        logging.StreamHandler()
    ]
)

# tune for your hazard keywords
QUERY = "flood OR flooded OR #flooding near:India"
OUTFILE = "tweets_stream.jsonl"


def stream_once(query, max_results=200, max_retries=3):
    """
    Attempt to scrape tweets with error handling and retry logic
    """
    for attempt in range(max_retries):
        try:
            with open(OUTFILE, "a", encoding="utf-8") as f:
                logging.info(f"Starting scraping attempt {attempt + 1}/{max_retries} for query: {query}")
                
                tweets_collected = 0
                scraper = sntwitter.TwitterSearchScraper(query)
                
                for i, tweet in enumerate(scraper.get_items()):
                    data = {
                        "id": tweet.id,
                        "datetime": tweet.date.isoformat(),
                        "user": tweet.user.username,
                        "content": tweet.content,
                        "place": tweet.place.name if tweet.place else None,
                        "lang": tweet.lang
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    tweets_collected += 1
                    
                    if i >= max_results - 1:
                        break
                    
                    # Add small delay between tweets to avoid rate limiting
                    time.sleep(0.1)
                
                logging.info(f"Successfully collected {tweets_collected} tweets")
                return tweets_collected
                
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) * 60 + random.uniform(0, 30)
                logging.info(f"Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
            else:
                logging.error("All retry attempts failed")
                return 0
    
    return 0


if __name__ == "__main__":
    logging.info("Starting Twitter monitoring script...")
    logging.info(f"Query: {QUERY}")
    logging.info(f"Output file: {OUTFILE}")
    
    # Enhanced polling loop with better error handling
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        try:
            start_time = datetime.now()
            tweets_collected = stream_once(QUERY, max_results=100)  # Reduced batch size
            
            if tweets_collected > 0:
                consecutive_failures = 0
                logging.info(f"Cycle completed successfully. Collected {tweets_collected} tweets.")
                # Normal sleep time
                sleep_time = 300  # 5 minutes between successful runs
            else:
                consecutive_failures += 1
                logging.warning(f"No tweets collected. Consecutive failures: {consecutive_failures}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logging.error(f"Too many consecutive failures ({consecutive_failures}). Entering extended cooldown.")
                    sleep_time = 1800  # 30 minutes cooldown
                    consecutive_failures = 0  # Reset counter
                else:
                    sleep_time = 600  # 10 minutes between failed attempts
            
            logging.info(f"Waiting {sleep_time/60:.1f} minutes before next cycle...")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logging.info("Script interrupted by user. Exiting gracefully...")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {str(e)}")
            time.sleep(600)  # 10 minute cooldown on unexpected errors
