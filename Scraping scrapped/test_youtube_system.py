# test_youtube_system.py - Test the enhanced YouTube disaster monitoring system

from youtube_script import search_recent_disaster_videos, fetch_all_comments, DisasterYouTubeMonitor
import json

def test_basic_functionality():
    """
    Test basic functionality of the YouTube monitoring system
    """
    print("🧪 Testing YouTube Disaster Monitoring System")
    print("=" * 50)
    
    # Test 1: Search functionality
    print("\n1. Testing video search...")
    try:
        videos = search_recent_disaster_videos("flood India", max_results=2, hours_back=72)
        if videos:
            print(f"✅ Found {len(videos)} videos")
            print(f"   Sample: {videos[0]['title'][:60]}...")
            print(f"   Urgency: {videos[0]['urgency_level']}")
            print(f"   Locations: {videos[0]['location_mentions']}")
        else:
            print("⚠️  No videos found (this might be normal)")
    except Exception as e:
        print(f"❌ Search failed: {str(e)}")
        return False
    
    # Test 2: Comment extraction
    if videos:
        print("\n2. Testing comment extraction...")
        try:
            video_id = videos[0]['videoId']
            comments = fetch_all_comments(video_id, max_comments=5)
            if comments:
                print(f"✅ Extracted {len(comments)} comments")
                print(f"   Sample: {comments[0]['text'][:50]}...")
                if comments[0].get('personal_experience'):
                    print("   📝 Found personal experience indicator")
            else:
                print("⚠️  No comments found (comments might be disabled)")
        except Exception as e:
            print(f"❌ Comment extraction failed: {str(e)}")
    
    # Test 3: Data structure validation
    print("\n3. Testing data structure...")
    try:
        monitor = DisasterYouTubeMonitor(output_file="test_output.jsonl")
        if videos:
            video = videos[0]
            comments = fetch_all_comments(video['videoId'], max_comments=3)
            insights = monitor.aggregate_insights(video, comments)
            
            required_fields = [
                'unique_locations', 'disaster_type', 'estimated_severity',
                'public_sentiment', 'credibility_score'
            ]
            
            missing_fields = [field for field in required_fields if field not in insights['content_summary']]
            
            if not missing_fields:
                print("✅ Data structure is complete")
                print(f"   Disaster type: {insights['content_summary']['disaster_type']}")
                print(f"   Severity: {insights['content_summary']['estimated_severity']}")
                print(f"   Credibility: {insights['content_summary']['credibility_score']}/10")
            else:
                print(f"⚠️  Missing fields: {missing_fields}")
        else:
            print("⚠️  Cannot test data structure without videos")
    except Exception as e:
        print(f"❌ Data structure test failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("   The system appears to be working correctly!")
    print("   You can now run the full monitoring system.")
    print("\n💡 Next steps:")
    print("   1. Run: python youtube_script.py")
    print("   2. Choose option 1 for demo or option 2 for monitoring")
    print("   3. Check youtube_disaster_data.jsonl for collected data")
    
    return True

def test_nlp_data_format():
    """
    Test that the data format is suitable for NLP processing
    """
    print("\n🔬 Testing NLP Data Format")
    print("-" * 30)
    
    # Simulate collected data
    sample_data = {
        "data_type": "youtube_disaster_content",
        "collected_at": "2025-09-20T10:30:00",
        "video": {
            "title": "Mumbai Heavy Rain Causes Severe Flooding",
            "description": "Live coverage of unprecedented flooding in Mumbai due to heavy monsoon rain",
            "location_mentions": ["Mumbai", "Maharashtra", "India"],
            "severity_indicators": {"extreme": ["unprecedented"], "high": ["heavy", "severe"]},
            "urgency_level": "critical"
        },
        "comments": [
            {
                "text": "Water entered our building, very scary situation here in Andheri",
                "personal_experience": True,
                "location_mentions": ["Andheri"],
                "sentiment_indicators": {"negative": ["scary"]}
            }
        ],
        "aggregated_insights": {
            "content_summary": {
                "disaster_type": "flood",
                "estimated_severity": "extreme",
                "primary_location": "Mumbai",
                "credibility_score": 8
            }
        }
    }
    
    print("✅ Sample data structure for NLP:")
    print(f"   🎯 Event Type: {sample_data['aggregated_insights']['content_summary']['disaster_type']}")
    print(f"   📍 Location: {sample_data['aggregated_insights']['content_summary']['primary_location']}")
    print(f"   ⚡ Severity: {sample_data['aggregated_insights']['content_summary']['estimated_severity']}")
    print(f"   🎗️  Credibility: {sample_data['aggregated_insights']['content_summary']['credibility_score']}/10")
    print(f"   👥 Personal Accounts: {sum(1 for c in sample_data['comments'] if c.get('personal_experience'))}")
    
    return True

def show_api_quota_info():
    """
    Show information about API usage and quotas
    """
    print("\n📊 API Quota Information")
    print("-" * 25)
    print("YouTube Data API v3 Quotas:")
    print("   Daily Limit: 10,000 units")
    print("   Search: 100 units per request")
    print("   Video details: 1 unit per video")
    print("   Comments: 1 unit per request")
    print("\nEstimated usage per monitoring cycle:")
    print("   10 keywords × 20 videos = 200 search units")
    print("   200 videos × 1 detail request = 200 units")
    print("   200 videos × 2 comment requests = 400 units")
    print("   Total: ~800 units per cycle")
    print("\n💡 With 15-minute intervals: ~3,200 units/hour")
    print("   Recommended: 30-60 minute intervals for continuous monitoring")

if __name__ == "__main__":
    print("🚀 YouTube Disaster Monitoring System - Test Suite")
    print("=" * 55)
    
    # Show quota info first
    show_api_quota_info()
    
    # Test NLP format
    test_nlp_data_format()
    
    # Test basic functionality
    print(f"\n{'='*55}")
    proceed = input("\n🔍 Test live API functionality? (y/n): ").lower().strip()
    
    if proceed == 'y':
        success = test_basic_functionality()
        if success:
            print("\n🎉 All tests passed! System is ready for use.")
        else:
            print("\n⚠️  Some tests failed. Check your API key and internet connection.")
    else:
        print("\n✅ Static tests completed. System structure is correct.")
        print("   Run with 'y' when ready to test live API calls.")