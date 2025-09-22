#!/usr/bin/env python3
"""
Redis Consumer for Disaster Social Media Simulator
==================================================

Consumes social media posts from Redis stream and processes them.
Can be used for real-time analysis, storage, or forwarding to other systems.

Author: Disaster Analytics Team
Date: September 2025
"""

import json
import time
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional
import argparse

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Error: Redis not installed. Install with: pip install redis")
    sys.exit(1)


class DisasterPostConsumer:
    """
    Consumer for processing disaster-related social media posts from Redis stream.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, detailed: bool = True):
        """Initialize the Redis consumer."""
        self.host = host
        self.port = port
        self.db = db
        self.detailed = detailed
        self.redis_client = None
        self.stream_name = "disaster_posts"
        self.consumer_group = "disaster_analytics"
        self.consumer_name = f"consumer_{int(time.time())}"
        self.running = True
        
        # Statistics
        self.posts_processed = 0
        self.posts_by_platform = {}
        self.posts_by_type = {}
        self.start_time = time.time()
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\nShutdown signal received. Stopping consumer...")
        self.running = False
    
    def connect(self) -> bool:
        """Connect to Redis and setup consumer group."""
        try:
            self.redis_client = redis.Redis(
                host=self.host, 
                port=self.port, 
                db=self.db, 
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            print(f"✓ Connected to Redis at {self.host}:{self.port}")
            
            # Create consumer group if it doesn't exist
            try:
                self.redis_client.xgroup_create(
                    self.stream_name, 
                    self.consumer_group, 
                    id='0', 
                    mkstream=True
                )
                print(f"✓ Created consumer group: {self.consumer_group}")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    print(f"✓ Consumer group already exists: {self.consumer_group}")
                else:
                    raise
            
            return True
            
        except redis.ConnectionError:
            print(f"✗ Failed to connect to Redis at {self.host}:{self.port}")
            print("Make sure Redis server is running.")
            return False
        except Exception as e:
            print(f"✗ Redis setup error: {e}")
            return False
    
    def process_post(self, post_data: Dict) -> None:
        """Process a single post. Override this method for custom processing."""
        try:
            # Parse the post data
            post = json.loads(post_data.get('post_data', '{}'))
            platform = post_data.get('platform', 'unknown')
            post_type = post_data.get('post_type', 'unknown')
            timestamp = post_data.get('timestamp', datetime.now().isoformat())
            
            # Update statistics
            self.posts_processed += 1
            self.posts_by_platform[platform] = self.posts_by_platform.get(platform, 0) + 1
            self.posts_by_type[post_type] = self.posts_by_type.get(post_type, 0) + 1
            
            # Show header
            print(f"\n{'='*80}")
            print(f"📱 POST #{self.posts_processed:04d} | {platform.upper()} | {post_type.upper()}")
            print(f"⏰ Received at: {timestamp}")
            print(f"{'='*80}")
            
            if self.detailed:
                # Show complete JSON with pretty formatting
                print("📄 FULL POST DATA:")
                print(json.dumps(post, indent=2, ensure_ascii=False))
            else:
                # Show summary (original behavior)
                print(f"👤 User: {post.get('user', 'N/A')}")
                print(f"🌐 Lang: {post.get('lang', 'N/A')}")
                
                # Print content based on platform
                if platform == 'twitter':
                    print(f"💬 Text: {post.get('text', '')[:100]}...")
                    print(f"📊 Likes: {post.get('likes', 0)} | Retweets: {post.get('retweets', 0)}")
                elif platform == 'news':
                    print(f"📰 Headline: {post.get('headline', '')[:100]}...")
                    print(f"✍️  Author: {post.get('author', 'N/A')}")
                elif platform == 'youtube':
                    print(f"🎥 Title: {post.get('title', '')[:100]}...")
                    print(f"👁️  Views: {post.get('views', 0)} | Likes: {post.get('likes', 0)}")
                elif platform in ['facebook', 'instagram']:
                    content = post.get('content', '') or post.get('caption', '')
                    print(f"📝 Content: {content[:100]}...")
                    print(f"❤️  Likes: {post.get('likes', 0)} | Comments: {post.get('comments', 0)}")
            
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"❌ Error processing post: {e}")
            print(f"Raw data: {post_data}")
            print(f"{'='*80}")
    
    def print_statistics(self) -> None:
        """Print current processing statistics."""
        elapsed = time.time() - self.start_time
        rate = self.posts_processed / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("📊 CONSUMER STATISTICS")
        print("="*60)
        print(f"⏱️  Runtime: {elapsed:.1f} seconds")
        print(f"📨 Posts processed: {self.posts_processed}")
        print(f"📈 Processing rate: {rate:.2f} posts/second")
        print()
        print("📱 By Platform:")
        for platform, count in self.posts_by_platform.items():
            percentage = (count / self.posts_processed) * 100 if self.posts_processed > 0 else 0
            print(f"   {platform:12}: {count:4d} ({percentage:5.1f}%)")
        print()
        print("📂 By Type:")
        for post_type, count in self.posts_by_type.items():
            percentage = (count / self.posts_processed) * 100 if self.posts_processed > 0 else 0
            print(f"   {post_type:12}: {count:4d} ({percentage:5.1f}%)")
        print("="*60)
    
    def consume(self, timeout: int = 1000, stats_interval: int = 30) -> None:
        """Start consuming messages from Redis stream."""
        if not self.connect():
            return
        
        print(f"🚀 Starting consumer: {self.consumer_name}")
        print(f"📡 Stream: {self.stream_name}")
        print(f"👥 Group: {self.consumer_group}")
        print(f"⏰ Stats interval: {stats_interval}s")
        print("Press Ctrl+C to stop...")
        print("-" * 80)
        
        last_stats_time = time.time()
        
        try:
            while self.running:
                try:
                    # Read messages from stream
                    messages = self.redis_client.xreadgroup(
                        self.consumer_group,
                        self.consumer_name,
                        {self.stream_name: '>'},
                        count=1,
                        block=timeout
                    )
                    
                    # Process messages
                    for stream, msgs in messages:
                        for msg_id, fields in msgs:
                            self.process_post(fields)
                            
                            # Acknowledge message
                            self.redis_client.xack(self.stream_name, self.consumer_group, msg_id)
                    
                    # Print periodic statistics
                    if time.time() - last_stats_time >= stats_interval:
                        self.print_statistics()
                        last_stats_time = time.time()
                        
                except redis.ConnectionError:
                    print("Lost connection to Redis. Attempting to reconnect...")
                    time.sleep(5)
                    if not self.connect():
                        print("Failed to reconnect. Exiting...")
                        break
                except Exception as e:
                    print(f"Error reading from stream: {e}")
                    time.sleep(1)
        
        finally:
            self.print_statistics()
            if self.redis_client:
                self.redis_client.close()
            print("✅ Consumer stopped.")


def main():
    """Main function to run the consumer."""
    parser = argparse.ArgumentParser(description='Redis Consumer for Disaster Social Media Posts')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Redis host (default: localhost)')
    parser.add_argument('--port', type=int, default=6379,
                       help='Redis port (default: 6379)')
    parser.add_argument('--db', type=int, default=0,
                       help='Redis database (default: 0)')
    parser.add_argument('--timeout', type=int, default=1000,
                       help='Read timeout in milliseconds (default: 1000)')
    parser.add_argument('--stats', type=int, default=30,
                       help='Statistics interval in seconds (default: 30)')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary instead of full JSON (default: show full JSON)')
    
    args = parser.parse_args()
    
    # Create and start consumer
    consumer = DisasterPostConsumer(
        host=args.host, 
        port=args.port, 
        db=args.db, 
        detailed=not args.summary  # detailed=True unless --summary is specified
    )
    consumer.consume(timeout=args.timeout, stats_interval=args.stats)


if __name__ == "__main__":
    main()