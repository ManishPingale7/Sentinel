#!/usr/bin/env python3
"""
Test script for enhanced YouTube data collection with tags, captions, and no thumbnails
"""

import sys
import json
from youtube_script import (
    extract_comprehensive_tags, 
    extract_video_transcription_info,
    extract_video_metadata,
    search_recent_disaster_videos
)

def test_tag_extraction():
    """Test the comprehensive tag extraction function"""
    print("🏷️  Testing comprehensive tag extraction...")
    
    # Sample video data
    title = "Mumbai Floods 2024: Heavy Rain Causes Massive Flooding in Maharashtra #MumbaiFloods #Emergency"
    description = """
    Breaking: Heavy rainfall in Mumbai has caused severe flooding across multiple areas including Bandra, Andheri, and Dadar. 
    Emergency services are on high alert. Rescue operations underway in affected areas.
    
    Location: Mumbai, Maharashtra, India
    Weather Alert: Extreme rainfall warning
    
    Tags: #Weather #Disaster #Mumbai #Maharashtra #India #Rain #Flood #Emergency #Rescue
    Contact: @MumbaiPolice @BMCofficial
    
    cyclone warning evacuation flood damage heavy rain storm surge coastal erosion
    """
    
    youtube_tags = ["Mumbai", "Weather", "News", "Emergency", "Live", "Breaking"]
    
    # Test tag extraction
    result = extract_comprehensive_tags(title, description, youtube_tags)
    
    print(f"  ✅ Total tags extracted: {result['total_tag_count']}")
    print(f"  📍 Location mentions: {len(result['detailed_tags']['location_mentions'])}")
    print(f"  🚨 Disaster keywords: {len(result['detailed_tags']['disaster_keywords'])}")
    print(f"  #️⃣ Hashtags found: {len(result['detailed_tags']['hashtags'])}")
    print(f"  📺 YouTube tags: {len(result['detailed_tags']['youtube_tags'])}")
    
    print("\n  📊 Sample tags by category:")
    for category, tags in result['detailed_tags'].items():
        if tags:
            print(f"    {category}: {tags[:3]}{'...' if len(tags) > 3 else ''}")
    
    print(f"\n  🔗 Unified tags: {result['unified_tags'][:10]}{'...' if len(result['unified_tags']) > 10 else ''}")
    
    return True

def test_captions_info():
    """Test caption detection (without actual download)"""
    print("\n📺 Testing caption detection...")
    
    try:
        # Search for a recent video to test captions
        videos = search_recent_disaster_videos("news mumbai", max_results=2, hours_back=72)
        
        if videos:
            video = videos[0]
            print(f"  Testing with video: {video['title'][:60]}...")
            
            # Test transcription info
            transcription_info = extract_video_transcription_info(video['videoId'])
            
            print(f"  ✅ Captions available: {transcription_info.get('captions_available', False)}")
            print(f"  🎯 Caption status: {transcription_info.get('caption_status', 'unknown')}")
            
            if transcription_info.get('caption_tracks'):
                print(f"  📝 Caption tracks found: {len(transcription_info['caption_tracks'])}")
                for track in transcription_info['caption_tracks']:
                    print(f"    - Language: {track.get('language', 'unknown')}")
                    print(f"    - Auto-generated: {track.get('is_auto_generated', False)}")
            else:
                print("  ℹ️  No caption tracks found")
                
            return True
        else:
            print("  ⚠️  No videos found for testing captions")
            return False
            
    except Exception as e:
        print(f"  ❌ Caption test failed: {str(e)}")
        return False

def test_enhanced_metadata():
    """Test the complete enhanced metadata extraction"""
    print("\n🎯 Testing enhanced metadata extraction...")
    
    try:
        # Search for a test video
        videos = search_recent_disaster_videos("weather news", max_results=1, hours_back=72)
        
        if videos:
            video_item = videos[0]
            print(f"  Testing with: {video_item['title'][:50]}...")
            
            # Extract enhanced metadata
            metadata = extract_video_metadata(video_item)
            
            print("  ✅ Enhanced metadata extracted successfully!")
            
            # Check new features
            print(f"  🏷️  Tags structure: {type(metadata.get('tags', {}))}")
            if isinstance(metadata.get('tags'), dict):
                print(f"    - Detailed tags: {len(metadata['tags'].get('detailed_tags', {}))}")
                print(f"    - Unified tags: {len(metadata['tags'].get('unified_tags', []))}")
            
            print(f"  📺 Transcription info: {metadata.get('transcription', {}).get('captions_available', False)}")
            
            # Check thumbnails are removed
            has_thumbnails = (
                'thumbnails' in metadata or 
                any('thumbnails' in str(v) for v in metadata.values() if isinstance(v, dict))
            )
            print(f"  🚫 Thumbnails removed: {not has_thumbnails}")
            
            # Save test output
            with open("test_enhanced_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"  💾 Test data saved to test_enhanced_metadata.json")
            
            return True
        else:
            print("  ⚠️  No videos found for metadata testing")
            return False
            
    except Exception as e:
        print(f"  ❌ Enhanced metadata test failed: {str(e)}")
        return False

def test_no_thumbnails():
    """Verify that thumbnails are completely removed"""
    print("\n🚫 Testing thumbnail removal...")
    
    try:
        with open("test_enhanced_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Recursively check for thumbnail references
        def check_thumbnails(obj, path=""):
            thumbnail_found = False
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if 'thumbnail' in key.lower():
                        print(f"  ❌ Found thumbnail reference at: {path}.{key}")
                        thumbnail_found = True
                    if check_thumbnails(value, f"{path}.{key}"):
                        thumbnail_found = True
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if check_thumbnails(item, f"{path}[{i}]"):
                        thumbnail_found = True
            elif isinstance(obj, str):
                if 'thumbnail' in obj.lower() and 'youtube.com' in obj.lower():
                    print(f"  ❌ Found thumbnail URL at: {path}")
                    thumbnail_found = True
            
            return thumbnail_found
        
        if not check_thumbnails(metadata):
            print("  ✅ No thumbnail references found - successfully removed!")
            return True
        else:
            print("  ❌ Some thumbnail references still exist")
            return False
            
    except Exception as e:
        print(f"  ❌ Thumbnail check failed: {str(e)}")
        return False

def main():
    """Run all enhancement tests"""
    print("YouTube Enhancement Tests")
    print("=" * 50)
    
    tests = [
        ("Tag Extraction", test_tag_extraction),
        ("Caption Detection", test_captions_info),
        ("Enhanced Metadata", test_enhanced_metadata),
        ("Thumbnail Removal", test_no_thumbnails)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n💥 {test_name}: ERROR - {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All enhancements working correctly!")
    else:
        print("⚠️  Some enhancements need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)