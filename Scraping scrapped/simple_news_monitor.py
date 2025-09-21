# simple_news_monitor.py - Works with packages you already have
import json
import time
import logging
import requests
from datetime import datetime
import re
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_monitor.log'),
        logging.StreamHandler()
    ]
)

class SimpleNewsMonitor:
    def __init__(self):
        self.outfile = "flood_news.jsonl"
        self.flood_keywords = [
            'flood', 'flooded', 'flooding', 'floods', 'inundated', 'waterlogged',
            'deluge', 'submerg', 'overflow', 'heavy rain', 'monsoon', 'cyclone'
        ]
        
        # News websites to scrape (simplified approach)
        self.news_sources = [
            {
                'name': 'Times of India',
                'search_url': 'https://timesofindia.indiatimes.com/topic/flood/news',
                'base_url': 'https://timesofindia.indiatimes.com'
            },
            {
                'name': 'The Hindu',
                'search_url': 'https://www.thehindu.com/tag/flood/',
                'base_url': 'https://www.thehindu.com'
            }
        ]
    
    def is_flood_related(self, text):
        """Check if text contains flood-related keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.flood_keywords)
    
    def scrape_google_news(self):
        """Scrape Google News for flood-related articles in India"""
        try:
            query = "flood India news"
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse RSS-like content manually since we might not have feedparser
            soup = BeautifulSoup(response.content, 'xml' if 'xml' in response.headers.get('content-type', '') else 'html.parser')
            
            items = soup.find_all(['item', 'entry'])
            collected = 0
            
            with open(self.outfile, "a", encoding="utf-8") as f:
                for item in items[:15]:  # Limit to recent items
                    try:
                        title_elem = item.find(['title'])
                        desc_elem = item.find(['description', 'summary'])
                        link_elem = item.find(['link', 'guid'])
                        
                        title = title_elem.get_text() if title_elem else ''
                        description = desc_elem.get_text() if desc_elem else ''
                        link = link_elem.get_text() if link_elem else ''
                        
                        if self.is_flood_related(f"{title} {description}"):
                            news_data = {
                                "id": f"google_news_{collected}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                "datetime": datetime.now().isoformat(),
                                "source": "Google News",
                                "title": title,
                                "content": description,
                                "link": link,
                                "category": "news_aggregator",
                                "type": "flood_news",
                                "scraped_at": datetime.now().isoformat()
                            }
                            
                            f.write(json.dumps(news_data, ensure_ascii=False) + "\n")
                            collected += 1
                    
                    except Exception as e:
                        logging.warning(f"Error processing news item: {str(e)}")
                        continue
            
            return collected
            
        except Exception as e:
            logging.error(f"Error scraping Google News: {str(e)}")
            return 0
    
    def scrape_weather_updates(self):
        """Scrape weather-related updates"""
        try:
            # AccuWeather India weather alerts (simplified)
            url = "https://www.accuweather.com/en/in/india-weather"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            collected = 0
            
            # Look for weather alerts or warnings
            alert_elements = soup.find_all(['div', 'p', 'span'], 
                                         class_=re.compile(r'alert|warning|weather', re.I))
            
            with open(self.outfile, "a", encoding="utf-8") as f:
                for element in alert_elements[:10]:
                    text = element.get_text().strip()
                    if len(text) > 30 and self.is_flood_related(text):
                        weather_data = {
                            "id": f"weather_alert_{collected}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            "datetime": datetime.now().isoformat(),
                            "source": "AccuWeather India",
                            "title": "Weather Alert",
                            "content": text,
                            "link": url,
                            "category": "weather",
                            "type": "weather_alert",
                            "scraped_at": datetime.now().isoformat()
                        }
                        
                        f.write(json.dumps(weather_data, ensure_ascii=False) + "\n")
                        collected += 1
            
            return collected
            
        except Exception as e:
            logging.error(f"Error scraping weather updates: {str(e)}")
            return 0
    
    def collect_disaster_alerts(self):
        """Collect from disaster management websites"""
        try:
            # National Disaster Management Authority (simplified)
            url = "https://ndma.gov.in/en/media-public-awareness/advertisement.html"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            collected = 0
            
            # Look for disaster-related content
            content_elements = soup.find_all(['h2', 'h3', 'p', 'div'], 
                                           text=re.compile(r'flood|disaster|alert|warning', re.I))
            
            with open(self.outfile, "a", encoding="utf-8") as f:
                for element in content_elements[:5]:
                    text = element.get_text().strip()
                    if len(text) > 20 and self.is_flood_related(text):
                        alert_data = {
                            "id": f"ndma_alert_{collected}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            "datetime": datetime.now().isoformat(),
                            "source": "National Disaster Management Authority",
                            "title": "Disaster Alert",
                            "content": text,
                            "link": url,
                            "category": "official",
                            "type": "disaster_alert",
                            "scraped_at": datetime.now().isoformat()
                        }
                        
                        f.write(json.dumps(alert_data, ensure_ascii=False) + "\n")
                        collected += 1
            
            return collected
            
        except Exception as e:
            logging.error(f"Error collecting disaster alerts: {str(e)}")
            return 0
    
    def run_monitoring_cycle(self):
        """Run one complete monitoring cycle"""
        logging.info("Starting flood monitoring cycle...")
        
        total_collected = 0
        
        # Collect from Google News
        try:
            google_collected = self.scrape_google_news()
            total_collected += google_collected
            logging.info(f"Collected {google_collected} items from Google News")
            time.sleep(5)  # Be respectful
        except Exception as e:
            logging.error(f"Google News collection failed: {str(e)}")
        
        # Collect weather updates
        try:
            weather_collected = self.scrape_weather_updates()
            total_collected += weather_collected
            logging.info(f"Collected {weather_collected} weather alerts")
            time.sleep(5)
        except Exception as e:
            logging.error(f"Weather collection failed: {str(e)}")
        
        # Collect disaster alerts
        try:
            disaster_collected = self.collect_disaster_alerts()
            total_collected += disaster_collected
            logging.info(f"Collected {disaster_collected} disaster alerts")
            time.sleep(5)
        except Exception as e:
            logging.error(f"Disaster alerts collection failed: {str(e)}")
        
        logging.info(f"Monitoring cycle completed. Total items: {total_collected}")
        return total_collected

def main():
    monitor = SimpleNewsMonitor()
    
    logging.info("Starting Simple Flood News Monitor...")
    logging.info("This system monitors flood-related news and alerts for India")
    logging.info("Using web scraping with packages you already have installed")
    
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while True:
        try:
            start_time = datetime.now()
            collected = monitor.run_monitoring_cycle()
            
            if collected > 0:
                consecutive_failures = 0
                logging.info(f"Successfully collected {collected} flood-related items")
                sleep_time = 2700  # 45 minutes between successful runs
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
            logging.info("Monitor interrupted by user. Exiting gracefully...")
            break
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {str(e)}")
            time.sleep(3600)  # 1 hour cooldown on unexpected errors

if __name__ == "__main__":
    main()