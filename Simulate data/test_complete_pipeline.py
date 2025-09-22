#!/usr/bin/env python3
"""
Complete Pipeline Test Script
Tests the full disaster monitoring pipeline: Simulator -> Redis -> NLP Engine -> Firebase
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def check_redis_running():
    """Check if Redis server is running."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        return True
    except:
        return False

def start_producer(duration=30):
    """Start the post producer."""
    print("🚀 Starting post producer...")
    cmd = [
        "G:/Sentinel/myenv/Scripts/python.exe", 
        "simulator.py", 
        "-r",  # Redis mode
        "--duration", str(duration)
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def start_consumer_with_nlp(gemini_key, firebase_cred=None):
    """Start the NLP-enabled consumer."""
    print("🤖 Starting NLP consumer...")
    cmd = [
        "G:/Sentinel/myenv/Scripts/python.exe", 
        "enhanced_consumer.py",
        "--gemini-key", gemini_key
    ]
    if firebase_cred and Path(firebase_cred).exists():
        cmd.extend(["--firebase-cred", firebase_cred])
        print(f"   📊 Firebase enabled: {firebase_cred}")
    else:
        print("   ⚠️  Firebase disabled (no credentials)")
    
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    """Run the complete pipeline test."""
    print("="*60)
    print("🛡️  SENTINEL DISASTER MONITORING PIPELINE TEST")
    print("="*60)
    
    # Configuration
    gemini_key = "AIzaSyDEEkyXceYLFknvkoVKYpIVnLWfSRz3wEY"
    firebase_cred = "sentinel-ed93e-firebase-adminsdk-fbsvc-996cb7c18e.json"
    test_duration = 30  # seconds
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    if not check_redis_running():
        print("❌ Redis server not running!")
        print("   Start Redis with: docker run --name redis-server -d -p 6379:6379 redis:latest")
        return False
    print("✅ Redis server is running")
    
    if not Path(firebase_cred).exists():
        print(f"⚠️  Firebase credentials not found: {firebase_cred}")
        print("   System will work without Firebase storage")
    else:
        print("✅ Firebase credentials found")
    
    print(f"\n🚀 Starting {test_duration}-second pipeline test...")
    print("   This will generate posts and process them through the NLP engine")
    
    # Start consumer first
    consumer_proc = start_consumer_with_nlp(gemini_key, firebase_cred)
    time.sleep(2)  # Give consumer time to start
    
    # Start producer
    producer_proc = start_producer(test_duration)
    
    try:
        print(f"\n⏱️  Running for {test_duration} seconds...")
        print("   Press Ctrl+C to stop early")
        
        # Wait for producer to finish
        producer_proc.wait(timeout=test_duration + 10)
        
        print("\n📊 Producer finished. Giving consumer time to process remaining posts...")
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopping pipeline...")
    except subprocess.TimeoutExpired:
        print("\n⏰ Test duration completed")
    
    finally:
        # Clean shutdown
        print("🛑 Shutting down processes...")
        
        if producer_proc.poll() is None:
            producer_proc.terminate()
            producer_proc.wait(timeout=5)
        
        if consumer_proc.poll() is None:
            consumer_proc.terminate()
            consumer_proc.wait(timeout=5)
        
        print("✅ Pipeline test completed!")
        
        # Show some output
        print("\n📋 Producer output (last 10 lines):")
        if producer_proc.stdout:
            lines = producer_proc.stdout.read().decode().split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"   {line}")
        
        print("\n🤖 Consumer output (last 10 lines):")
        if consumer_proc.stdout:
            lines = consumer_proc.stdout.read().decode().split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"   {line}")

if __name__ == "__main__":
    main()