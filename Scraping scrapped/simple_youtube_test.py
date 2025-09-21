# simple_youtube_test.py - Simple test to validate YouTube API access

from googleapiclient.discovery import build
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_api_key():
    """Test YouTube API key with minimal request"""
    
    API_KEY = "AIzaSyCifSphkbSWJSM56Hq_feutq-fs7QcrLs0"
    
    print("🧪 Testing YouTube API Key")
    print("=" * 30)
    
    # Basic validation
    print(f"API Key: {API_KEY[:20]}...")
    print(f"Length: {len(API_KEY)} characters")
    print(f"Format: {'✅ Valid' if API_KEY.startswith('AIza') and len(API_KEY) == 39 else '❌ Invalid'}")
    
    try:
        # Initialize API
        youtube = build("youtube", "v3", developerKey=API_KEY)
        print("✅ API client initialized")
        
        # Test with simplest possible request
        print("\n🔍 Testing basic search...")
        request = youtube.search().list(
            q="test",
            part="id",
            maxResults=1,
            type="video"
        )
        
        response = request.execute()
        
        if response.get("items"):
            print("✅ Basic search successful")
            print(f"   Found {len(response['items'])} result(s)")
            
            # Test with India-specific search
            print("\n🇮🇳 Testing India-specific search...")
            india_request = youtube.search().list(
                q="India news",
                part="id,snippet", 
                maxResults=2,
                type="video"
            )
            
            india_response = india_request.execute()
            
            if india_response.get("items"):
                print("✅ India search successful")
                for item in india_response["items"]:
                    print(f"   - {item['snippet']['title'][:50]}...")
            else:
                print("⚠️  India search returned no results")
                
        else:
            print("⚠️  Search returned no results")
            
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        
        if "invalid" in str(e).lower():
            print("\n🔧 Possible solutions:")
            print("   1. Check API key is correct")
            print("   2. Ensure YouTube Data API v3 is enabled")
            print("   3. Verify billing is set up in Google Cloud Console")
            
        elif "quota" in str(e).lower():
            print("\n🔧 Quota exceeded:")
            print("   1. Wait for quota reset (daily)")
            print("   2. Check quota usage in Google Cloud Console")
            
        elif "forbidden" in str(e).lower():
            print("\n🔧 Access forbidden:")
            print("   1. Check API key permissions")
            print("   2. Verify API is enabled for your project")
            
        return False
    
    print("\n🎉 API validation successful!")
    return True

if __name__ == "__main__":
    success = test_api_key()
    
    if success:
        print("\n✅ Your API key is working!")
        print("   You can now run the full monitoring system.")
    else:
        print("\n❌ API key needs attention.")
        print("   Please fix the issues above and try again.")