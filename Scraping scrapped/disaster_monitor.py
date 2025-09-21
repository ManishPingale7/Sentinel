# disaster_monitor.py - Working alternative to Twitter scraping
import json
import time
import logging
import requests
from datetime import datetime, timedelta
import re
from urllib.parse import urljoin, urlparse
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('disaster_monitor.log'),
        logging.StreamHandler()
    ]
)

class DisasterMonitor:
    def __init__(self):
        self.outfile = "disaster_alerts.jsonl"
        self.flood_keywords = [
            'flood', 'flooded', 'flooding', 'floods', 'inundated', 'waterlogged',
            'deluge', 'submerg', 'overflow', 'rain', 'monsoon', 'cyclone',
            'hurricane', 'tsunami', 'storm surge', 'heavy rain'
        ]
        self.india_keywords = [
            'india', 'indian', 'mumbai', 'delhi', 'kolkata', 'chennai', 'bangalore',
            'hyderabad', 'pune', 'ahmedabad', 'kerala', 'maharashtra', 'gujarat',
            'bihar', 'west bengal', 'odisha', 'assam', 'goa', 'uttarakhand'
        ]
        
        # RSS feeds for Indian news sources
        self.rss_feeds = [
            {
                'name': 'Times of India',
                'url': 'https://timesofindia.indiatimes.com/rssfeeds/296589292.cms',
                'category': 'news'
            },
            {
                'name': 'The Hindu',
                'url': 'https://www.thehindu.com/news/national/?service=rss',
                'category': 'news'
            },
            {
                'name': 'NDTV',
                'url': 'https://feeds.feedburner.com/ndtvnews-latest',
                'category': 'news'
            },
            {
                'name': 'India Today',
                'url': 'https://www.indiatoday.in/rss/1206514',
                'category': 'news'
            },
            {
                'name': 'BBC India',
                'url': 'http://feeds.bbci.co.uk/news/world/asia/india/rss.xml',
                'category': 'international'
            },
            {
                'name': 'Reuters India',
                'url': 'https://feeds.reuters.com/reuters/INtopNews',
                'category': 'international'
            }
        ]
        
        # Weather and disaster specific sources
        self.weather_sources = [
            {
                'name': 'India Meteorological Department',
                'url': 'https://mausam.imd.gov.in/imd_latest/contents/all_warning.php',
                'type': 'web_scrape'
            }
        ]

    def is_flood_related(self, text):
        """Check if text contains flood-related keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.flood_keywords)
    
    def is_india_related(self, text):
        """Check if text is related to India"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.india_keywords)
    
    def process_rss_feeds(self):
        """Collect data from RSS feeds"""
        total_collected = 0
        
        for feed_info in self.rss_feeds:
            try:
                collected = self._process_single_rss(feed_info)
                total_collected += collected
                logging.info(f"Collected {collected} items from {feed_info['name']}")
                time.sleep(2)  # Be respectful to servers
            except Exception as e:
                logging.error(f"Error processing {feed_info['name']}: {str(e)}")
        
        return total_collected
    
    def _process_single_rss(self, feed_info):
        """Process a single RSS feed"""
        try:
            import feedparser
        except ImportError:
            logging.error("feedparser not installed. Please install: pip install feedparser")
            return 0
        
        try:
            headers = {
                'User-Agent': 'DisasterMonitor/1.0 (Educational Research)'
            }
            
            # Fetch the RSS feed
            response = requests.get(feed_info['url'], headers=headers, timeout=10)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            collected_count = 0
            
            with open(self.outfile, "a", encoding="utf-8") as f:
                for entry in feed.entries[:20]:  # Limit to recent entries
                    try:
                        # Get content
                        title = getattr(entry, 'title', '')
                        summary = getattr(entry, 'summary', '')
                        content = f"{title} {summary}"
                        
                        # Check if it's flood and India related
                        if self.is_flood_related(content) and self.is_india_related(content):
                            alert_data = {
                                "id": getattr(entry, 'id', entry.link),
                                "datetime": getattr(entry, 'published', datetime.now().isoformat()),
                                "source": feed_info['name'],
                                "title": title,
                                "content": summary,
                                "link": getattr(entry, 'link', ''),
                                "category": feed_info['category'],
                                "type": "flood_alert",
                                "scraped_at": datetime.now().isoformat()
                            }
                            
                            f.write(json.dumps(alert_data, ensure_ascii=False) + "\n")
                            collected_count += 1
                    
                    except Exception as e:
                        logging.warning(f"Error processing entry: {str(e)}")
                        continue
            
            return collected_count
            
        except Exception as e:
            logging.error(f"Error fetching RSS feed {feed_info['url']}: {str(e)}")
            return 0
    
    def collect_weather_warnings(self):
        """Collect from official weather sources"""
        collected = 0
        
        # This is a placeholder for web scraping weather warnings
        # You would implement specific scrapers for weather websites
        try:
            # Example: scrape IMD warnings (simplified)
            collected += self._scrape_imd_warnings()
        except Exception as e:
            logging.error(f"Error collecting weather warnings: {str(e)}")
        
        return collected
    
    def _scrape_imd_warnings(self):
        """Scrape India Meteorological Department warnings"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logging.error("BeautifulSoup not installed. Please install: pip install beautifulsoup4")
            return 0
        
        try:
            url = "https://mausam.imd.gov.in/imd_latest/contents/all_warning.php"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            warnings_collected = 0
            
            # Look for warning content (this would need to be customized based on actual site structure)
            warning_elements = soup.find_all(['div', 'p', 'td'], text=re.compile(r'flood|rain|cyclone', re.I))
            
            with open(self.outfile, "a", encoding="utf-8") as f:
                for element in warning_elements[:10]:  # Limit results
                    text = element.get_text().strip()
                    if len(text) > 50 and self.is_flood_related(text):
                        warning_data = {
                            "id": f"imd_warning_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{warnings_collected}",
                            "datetime": datetime.now().isoformat(),
                            "source": "India Meteorological Department",
                            "title": "Weather Warning",
                            "content": text,
                            "link": url,
                            "category": "official_warning",
                            "type": "weather_warning",
                            "scraped_at": datetime.now().isoformat()
                        }
                        
                        f.write(json.dumps(warning_data, ensure_ascii=False) + "\n")
                        warnings_collected += 1
            
            return warnings_collected
            
        except Exception as e:
            logging.error(f"Error scraping IMD warnings: {str(e)}")
            return 0
    
    def collect_from_news_apis(self):
        """Collect from news APIs if available"""
        # This would require API keys for services like NewsAPI
        # Placeholder for now
        return 0
    
    def run_collection_cycle(self):
        """Run one complete collection cycle"""
        logging.info("Starting disaster monitoring collection cycle...")
        
        total_collected = 0
        
        # Collect from RSS feeds
        rss_collected = self.process_rss_feeds()
        total_collected += rss_collected
        
        # Collect weather warnings
        weather_collected = self.collect_weather_warnings()
        total_collected += weather_collected
        
        # Collect from news APIs (if configured)
        api_collected = self.collect_from_news_apis()
        total_collected += api_collected
        
        logging.info(f"Collection cycle completed. Total items: {total_collected}")
        return total_collected

def main():
    monitor = DisasterMonitor()
    
    logging.info("Starting Disaster Monitoring System...")
    logging.info("This system monitors flood-related news and alerts for India")
    
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        try:
            start_time = datetime.now()
            collected = monitor.run_collection_cycle()
            
            if collected > 0:
                consecutive_failures = 0
                logging.info(f"Successfully collected {collected} disaster-related items")
                sleep_time = 1800  # 30 minutes between successful runs
            else:
                consecutive_failures += 1
                logging.warning(f"No items collected. Consecutive failures: {consecutive_failures}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logging.error("Too many consecutive failures. Extended cooldown.")
                    sleep_time = 7200  # 2 hours
                    consecutive_failures = 0
                else:
                    sleep_time = 3600  # 1 hour between failed attempts
            
            logging.info(f"Waiting {sleep_time/60:.1f} minutes before next cycle...")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logging.info("System interrupted by user. Exiting gracefully...")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {str(e)}")
            time.sleep(3600)  # 1 hour cooldown on unexpected errors

if __name__ == "__main__":
    main()