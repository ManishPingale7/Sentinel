#!/usr/bin/env python3
"""
Test script for raw data collection from YouTube
"""

import sys
import json
from youtube_script import DisasterYouTubeMonitor

def test_single_video_data():
    """Test comprehensive data collection for a single video"""
    print("Testing raw data collection...")
    
    try:
        # Initialize monitor
        monitor = DisasterYouTubeMonitor()
        
        # Test single keyword search
        from youtube_script import search_recent_disaster_videos, fetch_all_comments, aggregate_raw_data
        
        print("Searching for recent disaster videos...")
        videos = search_recent_disaster_videos("flood India", max_results=3, hours_back=24)
        
        if videos:
            print(f"Found {len(videos)} videos")
            
            # Test with first video
            video = videos[0]
            print(f"Testing with video: {video['title'][:50]}...")
            
            # Get comments
            comments = fetch_all_comments(video["videoId"], max_comments=10)
            print(f"Retrieved {len(comments)} comments")
            
            # Get complete raw data
            raw_data = aggregate_raw_data(video, comments, "flood India")
            
            # Display structure
            print("\n=== RAW DATA STRUCTURE ===")
            print("Main sections:")
            for key in raw_data.keys():
                if isinstance(raw_data[key], dict):
                    print(f"  {key}: {len(raw_data[key])} fields")
                elif isinstance(raw_data[key], list):
                    print(f"  {key}: {len(raw_data[key])} items")
                else:
                    print(f"  {key}: {type(raw_data[key]).__name__}")
            
            # Sample data points
            print(f"\nVideo Title: {raw_data['video_metadata']['title'][:80]}...")
            print(f"Channel: {raw_data['video_metadata']['channel_info']['title']}")
            print(f"Views: {raw_data['video_metadata']['statistics']['viewCount']}")
            print(f"Comments collected: {len(raw_data['all_comments'])}")
            print(f"Hashtags found: {len(raw_data['content_analysis']['hashtags'])}")
            print(f"Location mentions: {len(raw_data['content_analysis']['location_mentions'])}")
            
            # Save test data
            with open("test_raw_output.json", "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\nRaw data saved to test_raw_output.json")
            print("✅ Raw data collection test PASSED")
            
        else:
            print("❌ No videos found for testing")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False
    
    return True

def test_data_completeness():
    """Verify all required data fields are present"""
    try:
        with open("test_raw_output.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        required_sections = [
            'search_metadata',
            'video_metadata', 
            'all_comments',
            'content_analysis',
            'collection_timestamp'
        ]
        
        print("\n=== COMPLETENESS CHECK ===")
        for section in required_sections:
            if section in data:
                print(f"✅ {section}: Present")
            else:
                print(f"❌ {section}: Missing")
        
        # Check video metadata completeness
        video_meta = data.get('video_metadata', {})
        video_fields = ['title', 'description', 'channel_info', 'statistics', 'content_details']
        
        print("\nVideo metadata fields:")
        for field in video_fields:
            if field in video_meta:
                print(f"  ✅ {field}")
            else:
                print(f"  ❌ {field}")
        
        # Check comment structure
        if data.get('all_comments'):
            comment = data['all_comments'][0]
            comment_fields = ['id', 'text', 'author', 'metadata', 'engagement']
            
            print("\nComment structure:")
            for field in comment_fields:
                if field in comment:
                    print(f"  ✅ {field}")
                else:
                    print(f"  ❌ {field}")
        
        print("\n✅ Data completeness check completed")
        return True
        
    except Exception as e:
        print(f"❌ Completeness check failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("YouTube Raw Data Collection Test")
    print("=" * 40)
    
    # Test data collection
    if test_single_video_data():
        # Test completeness
        test_data_completeness()
    
    print("\nTest completed!")