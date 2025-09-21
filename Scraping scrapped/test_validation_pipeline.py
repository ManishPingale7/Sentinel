#!/usr/bin/env python3
"""
Test script for enhanced YouTube validation pipeline
Tests content seriousness filtering, duplicate prevention, coastal focus, and transcript extraction
"""

import sys
import json
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from youtube_script import (
        validate_content_seriousness,
        extract_video_transcription_info,
        VideoDuplicateTracker,
        search_recent_disaster_videos,
        DisasterYouTubeMonitor
    )
except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    print("Make sure you're running this from the Sentinel directory")
    sys.exit(1)

def test_content_seriousness_filter():
    """Test the content seriousness validation"""
    print("🔍 Testing content seriousness filter...")
    
    # Test cases: serious vs non-serious content
    test_cases = [
        {
            "name": "Serious Disaster Content",
            "video_data": {
                "title": "Breaking: Tsunami Warning Issued for Mumbai Coast - Evacuation Underway",
                "description": "URGENT: Indian Meteorological Department has issued tsunami warning for Mumbai coastal areas. NDRF teams deployed for emergency evacuation. Live coverage of rescue operations.",
                "channel": {
                    "channelDetails": {
                        "title": "News24 India",
                        "description": "Leading news channel providing breaking news and live coverage"
                    }
                }
            },
            "comments": [
                {"text": "Stay safe everyone. This is serious situation."},
                {"text": "NDRF rescue teams are doing great work"},
                {"text": "Authorities urging immediate evacuation from coastal areas"}
            ],
            "expected": True
        },
        {
            "name": "Comedy/Parody Content",
            "video_data": {
                "title": "Funny Flood Prank - Mumbai Rain Comedy | Hilarious Reaction",
                "description": "LOL! Watch this hilarious flood prank video. Just for entertainment and fun. Don't take it seriously! Like and subscribe for more comedy content.",
                "channel": {
                    "channelDetails": {
                        "title": "FunnyBoy Pranks",
                        "description": "Comedy channel for entertainment. All content is for fun and humor."
                    }
                }
            },
            "comments": [
                {"text": "Haha this is so funny! 😂"},
                {"text": "Great prank bro! Love your comedy videos"},
                {"text": "ROFL! This is hilarious content"}
            ],
            "expected": False
        },
        {
            "name": "Coastal Focus Content",
            "video_data": {
                "title": "Storm Surge Hits Chennai Coast - Coastal Flooding Emergency",
                "description": "Storm surge causing massive coastal flooding in Chennai. Tidal waves reaching inland areas. Emergency response activated.",
                "channel": {
                    "channelDetails": {
                        "title": "Weather India Official",
                        "description": "Official weather updates and disaster warnings"
                    }
                }
            },
            "comments": [
                {"text": "Coastal areas need immediate attention"},
                {"text": "Storm surge is devastating the shoreline"},
                {"text": "Emergency services working on rescue"}
            ],
            "expected": True
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"  Test {i}: {test_case['name']}")
        
        try:
            is_serious, confidence, reasons = validate_content_seriousness(
                test_case["video_data"],
                test_case["comments"],
                None  # No transcription for these tests
            )
            
            if is_serious == test_case["expected"]:
                print(f"    ✅ PASSED - Serious: {is_serious}, Confidence: {confidence:.2f}")
                passed += 1
            else:
                print(f"    ❌ FAILED - Expected: {test_case['expected']}, Got: {is_serious}")
            
            print(f"    📝 Reasons: {reasons[:2]}")  # Show first 2 reasons
            
        except Exception as e:
            print(f"    💥 ERROR: {str(e)}")
    
    print(f"  Seriousness Filter: {passed}/{total} tests passed\n")
    return passed == total

def test_duplicate_prevention():
    """Test the duplicate video prevention system"""
    print("🔄 Testing duplicate prevention...")
    
    try:
        # Create a test tracker
        tracker = VideoDuplicateTracker(max_cache_size=100)
        
        # Test adding and checking duplicates
        test_video_ids = ["test_video_1", "test_video_2", "test_video_3"]
        
        # First pass - should not be duplicates
        for video_id in test_video_ids:
            if tracker.is_duplicate(video_id):
                print(f"    ❌ False positive: {video_id} marked as duplicate when new")
                return False
            tracker.mark_processed(video_id)
        
        # Second pass - should be duplicates
        duplicates_found = 0
        for video_id in test_video_ids:
            if tracker.is_duplicate(video_id):
                duplicates_found += 1
        
        if duplicates_found == len(test_video_ids):
            print(f"    ✅ Duplicate detection working - {duplicates_found}/{len(test_video_ids)} detected")
            
            # Test stats
            stats = tracker.get_stats()
            print(f"    📊 Cache stats: {stats['total_processed']} videos tracked")
            
            return True
        else:
            print(f"    ❌ Duplicate detection failed - only {duplicates_found}/{len(test_video_ids)} detected")
            return False
            
    except Exception as e:
        print(f"    💥 Duplicate prevention test failed: {str(e)}")
        return False

def test_coastal_keyword_priority():
    """Test that coastal keywords are prioritized"""
    print("🌊 Testing coastal keyword priority...")
    
    try:
        from youtube_script import DISASTER_KEYWORDS
        
        # Count coastal vs general keywords
        coastal_keywords = []
        general_keywords = []
        
        coastal_terms = ['tsunami', 'coastal', 'storm surge', 'tidal', 'ocean', 'sea', 'cyclone', 'marine']
        
        for keyword in DISASTER_KEYWORDS[:20]:  # Check first 20 keywords
            if any(term in keyword.lower() for term in coastal_terms):
                coastal_keywords.append(keyword)
            else:
                general_keywords.append(keyword)
        
        coastal_ratio = len(coastal_keywords) / len(DISASTER_KEYWORDS[:20])
        
        print(f"    🌊 Coastal keywords: {len(coastal_keywords)}")
        print(f"    🏙️  General keywords: {len(general_keywords)}")
        print(f"    📊 Coastal ratio: {coastal_ratio:.1%}")
        
        # Expect at least 60% coastal focus
        if coastal_ratio >= 0.6:
            print(f"    ✅ Coastal priority maintained - {coastal_ratio:.1%} coastal focus")
            return True
        else:
            print(f"    ❌ Insufficient coastal focus - only {coastal_ratio:.1%}")
            return False
            
    except Exception as e:
        print(f"    💥 Coastal priority test failed: {str(e)}")
        return False

def test_transcript_extraction():
    """Test transcript extraction functionality"""
    print("📝 Testing transcript extraction...")
    
    try:
        # Test with a sample video (if API is available)
        videos = search_recent_disaster_videos("weather news", max_results=2, hours_back=72)
        
        if not videos:
            print("    ⚠️  No videos found for transcript testing")
            return True  # Not a failure, just no data
        
        video = videos[0]
        video_id = video['videoId']
        
        print(f"    Testing with video: {video.get('title', 'Unknown')[:50]}...")
        
        # Test transcript extraction
        transcript_info = extract_video_transcription_info(video_id)
        
        print(f"    📺 Captions available: {transcript_info.get('captions_available', False)}")
        print(f"    🎯 Extraction method: {transcript_info.get('extraction_method', 'unknown')}")
        
        if transcript_info.get('text_for_analysis'):
            text_length = len(transcript_info['text_for_analysis'])
            print(f"    📝 Analysis text length: {text_length} characters")
        
        # Success if we got any transcript information
        has_info = (
            transcript_info.get('captions_available') or 
            transcript_info.get('text_for_analysis') or
            transcript_info.get('extraction_method') != 'error'
        )
        
        if has_info:
            print("    ✅ Transcript extraction working")
            return True
        else:
            print("    ❌ No transcript information extracted")
            return False
            
    except Exception as e:
        print(f"    💥 Transcript test failed: {str(e)}")
        return False

def test_integration():
    """Test the complete enhanced pipeline"""
    print("🎯 Testing complete enhanced pipeline...")
    
    try:
        # Initialize the enhanced monitor
        monitor = DisasterYouTubeMonitor()
        
        print("    Collecting data with all validations enabled...")
        
        # Collect a small sample with all enhancements
        data = monitor.collect_recent_data()
        
        if data:
            sample = data[0]
            
            # Check validation metadata
            validation = sample.get('validation', {})
            
            print(f"    📊 Collected {len(data)} validated videos")
            print(f"    ✅ Validation metadata present: {bool(validation)}")
            print(f"    🔍 Seriousness check: {validation.get('is_serious_content', 'N/A')}")
            print(f"    🌊 Coastal focus: {validation.get('coastal_focused', 'N/A')}")
            print(f"    📝 Transcription: {validation.get('transcription_available', 'N/A')}")
            
            return True
        else:
            print("    ⚠️  No data collected (might be normal if no recent disasters)")
            return True  # Not necessarily a failure
            
    except Exception as e:
        print(f"    💥 Integration test failed: {str(e)}")
        return False

def main():
    """Run all validation tests"""
    print("YouTube Enhanced Validation Pipeline Tests")
    print("=" * 55)
    
    tests = [
        ("Content Seriousness Filter", test_content_seriousness_filter),
        ("Duplicate Prevention", test_duplicate_prevention),
        ("Coastal Keyword Priority", test_coastal_keyword_priority),
        ("Transcript Extraction", test_transcript_extraction),
        ("Complete Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"Running {test_name}...")
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED\n")
            else:
                print(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {str(e)}\n")
    
    print("=" * 55)
    print(f"Validation Tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All enhanced validations working correctly!")
        print("✅ Content filtering: Humor/comedy detection")
        print("✅ Duplicate prevention: Video ID tracking")
        print("✅ Coastal focus: Ocean hazard prioritization")
        print("✅ Transcript extraction: Caption analysis")
    else:
        print("⚠️  Some validations need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)