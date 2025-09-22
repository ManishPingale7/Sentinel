#!/usr/bin/env python3
"""
Enhanced Redis Consumer with NLP Processing
===========================================

Consumes social media posts from Redis stream and processes them through
the NLP engine for verification and information extraction.

Author: Disaster Analytics Team
Date: September 2025
"""

import json
import time
import signal
import sys
import os
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

# Import our NLP engine
from nlp_engine import DisasterNLPEngine


class EnhancedDisasterConsumer:
    """
    Enhanced consumer that processes disaster posts through NLP pipeline.
    """
    
    def __init__(self, 
                 gemini_api_key: str,
                 firebase_cred_path: Optional[str] = None,
                 host: str = "localhost", 
                 port: int = 6379, 
                 db: int = 0, 
                 detailed: bool = True):
        """Initialize the enhanced consumer."""
        self.host = host
        self.port = port
        self.db = db
        self.detailed = detailed
        self.redis_client = None
        self.stream_name = "disaster_posts"
        self.consumer_group = "disaster_analytics"
        self.consumer_name = f"enhanced_consumer_{int(time.time())}"
        self.running = True
        
        # Initialize NLP Engine
        self.nlp_engine = DisasterNLPEngine(
            gemini_api_key=gemini_api_key,
            firebase_cred_path=firebase_cred_path
        )
        
        # Statistics
        self.posts_received = 0
        self.posts_processed = 0
        self.posts_verified = 0
        self.posts_stored = 0
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
        """Process a single post through the NLP pipeline."""
        try:
            # Parse the post data
            post = json.loads(post_data.get('post_data', '{}'))
            platform = post_data.get('platform', 'unknown')
            post_type = post_data.get('post_type', 'unknown')
            timestamp = post_data.get('timestamp', datetime.now().isoformat())
            
            # Update basic statistics
            self.posts_received += 1
            self.posts_by_platform[platform] = self.posts_by_platform.get(platform, 0) + 1
            self.posts_by_type[post_type] = self.posts_by_type.get(post_type, 0) + 1
            
            # Show header
            print(f"\n{'='*80}")
            print(f"📱 POST #{self.posts_received:04d} | {platform.upper()} | {post_type.upper()}")
            print(f"⏰ Received at: {timestamp}")
            print(f"{'='*80}")
            
            if self.detailed:
                print("📄 ORIGINAL POST DATA:")
                print(json.dumps(post, indent=2, ensure_ascii=False))
                print(f"{'='*80}")
            
            # Process through NLP engine
            print("🤖 PROCESSING THROUGH NLP ENGINE...")
            nlp_result = self.nlp_engine.process_post(post)
            self.posts_processed += 1
            
            # Show results
            print("\n🔍 VERIFICATION RESULT:")
            if nlp_result.get('verification_result'):
                verification = nlp_result['verification_result']
                print(f"   ✅ Genuine: {nlp_result['genuine']}")
                print(f"   📊 Score: {verification.get('verification_score', 0):.2f}")
                print(f"   🎯 Confidence: {verification.get('confidence', 'unknown')}")
                print(f"   💭 Reasoning: {verification.get('reasoning', 'N/A')}")
            
            if nlp_result['genuine']:
                self.posts_verified += 1
                print("\n📋 EXTRACTED INFORMATION:")
                extracted = nlp_result.get('extracted_data', {})
                
                if extracted:
                    print(f"   🌪️  Disaster Type: {extracted.get('disaster_type', 'N/A')}")
                    print(f"   📍 Locations: {len(extracted.get('locations', []))} found")
                    for loc in extracted.get('locations', [])[:3]:  # Show first 3
                        print(f"      - {loc.get('name')} ({loc.get('type')})")
                    
                    print(f"   ⚠️  Impact Scale: {extracted.get('impact_scale', 'N/A')}")
                    print(f"   🚨 Urgency Level: {extracted.get('urgency_level', 'N/A')}")
                    print(f"   👥 Affected Population: {extracted.get('affected_population', 'N/A')}")
                    
                    if extracted.get('translated_content') and extracted.get('original_language') != 'English':
                        print(f"   🌐 Translated: {extracted['translated_content'][:100]}...")
                    
                    key_details = extracted.get('key_details', [])
                    if key_details:
                        print(f"   🔑 Key Details:")
                        for detail in key_details[:3]:  # Show first 3
                            print(f"      - {detail}")
                
                if nlp_result['stored']:
                    self.posts_stored += 1
                    print(f"\n💾 ✅ STORED TO FIREBASE")
                else:
                    print(f"\n💾 ❌ FIREBASE STORAGE FAILED")
            else:
                print("\n❌ POST DISCARDED (False alarm/prank)")
            
            if nlp_result.get('error'):
                print(f"\n⚠️ ERROR: {nlp_result['error']}")
            
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"❌ Error processing post: {e}")
            print(f"Raw data: {post_data}")
            print(f"{'='*80}")
    
    def print_statistics(self) -> None:
        """Print comprehensive processing statistics."""
        elapsed = time.time() - self.start_time
        rate = self.posts_received / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*80)
        print("📊 ENHANCED CONSUMER STATISTICS")
        print("="*80)
        print(f"⏱️  Runtime: {elapsed:.1f} seconds")
        print(f"📨 Posts received: {self.posts_received}")
        print(f"🤖 Posts processed by NLP: {self.posts_processed}")
        print(f"✅ Posts verified genuine: {self.posts_verified}")
        print(f"💾 Posts stored to Firebase: {self.posts_stored}")
        print(f"📈 Processing rate: {rate:.2f} posts/second")
        
        if self.posts_received > 0:
            print(f"🎯 Verification rate: {(self.posts_verified/self.posts_received)*100:.1f}%")
        if self.posts_verified > 0:
            print(f"💽 Storage success rate: {(self.posts_stored/self.posts_verified)*100:.1f}%")
        
        print()
        print("📱 By Platform:")
        for platform, count in self.posts_by_platform.items():
            percentage = (count / self.posts_received) * 100 if self.posts_received > 0 else 0
            print(f"   {platform:12}: {count:4d} ({percentage:5.1f}%)")
        
        print()
        print("📂 By Type:")
        for post_type, count in self.posts_by_type.items():
            percentage = (count / self.posts_received) * 100 if self.posts_received > 0 else 0
            print(f"   {post_type:12}: {count:4d} ({percentage:5.1f}%)")
        
        # Show NLP engine statistics
        print()
        self.nlp_engine.print_statistics()
        print("="*80)
    
    def consume(self, timeout: int = 1000, stats_interval: int = 60) -> None:
        """Start consuming messages from Redis stream."""
        if not self.connect():
            return
        
        print(f"🚀 Starting enhanced consumer: {self.consumer_name}")
        print(f"📡 Stream: {self.stream_name}")
        print(f"👥 Group: {self.consumer_group}")
        print(f"🤖 NLP Engine: Gemini API + Firebase")
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
            print("✅ Enhanced consumer stopped.")


def main():
    """Main function to run the enhanced consumer."""
    parser = argparse.ArgumentParser(description='Enhanced Redis Consumer with NLP Processing')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Redis host (default: localhost)')
    parser.add_argument('--port', type=int, default=6379,
                       help='Redis port (default: 6379)')
    parser.add_argument('--db', type=int, default=0,
                       help='Redis database (default: 0)')
    parser.add_argument('--timeout', type=int, default=1000,
                       help='Read timeout in milliseconds (default: 1000)')
    parser.add_argument('--stats', type=int, default=60,
                       help='Statistics interval in seconds (default: 60)')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary instead of full JSON (default: show full JSON)')
    parser.add_argument('--gemini-key', required=True,
                       help='Gemini API key for NLP processing')
    parser.add_argument('--firebase-cred', 
                       help='Firebase credentials JSON file path')
    
    args = parser.parse_args()
    
    # Validate Gemini API key
    if not args.gemini_key:
        print("Error: Gemini API key is required. Get it from https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    # Create and start enhanced consumer
    consumer = EnhancedDisasterConsumer(
        gemini_api_key=args.gemini_key,
        firebase_cred_path=args.firebase_cred,
        host=args.host, 
        port=args.port, 
        db=args.db, 
        detailed=not args.summary
    )
    consumer.consume(timeout=args.timeout, stats_interval=args.stats)


if __name__ == "__main__":
    main()