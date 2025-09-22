# 🛡️ Sentinel Disaster Monitoring System - Complete Setup Guide

## Prerequisites

### 1. Firebase Credentials
Download your Firebase service account credentials:
- Go to [Firebase Console](https://console.firebase.google.com/)
- Select project: `sentinel-ed93e`
- Navigate to: **Project Settings** → **Service Accounts**
- Click: **"Generate new private key"**
- Save as: `sentinel-ed93e-firebase-adminsdk-fbsvc-996cb7c18e.json`
- Place in: `g:\Sentinel\Simulate data\`

### 2. Redis Server
Start Redis server using Docker:
```bash
docker run --name redis-server -d -p 6379:6379 redis:latest
```

### 3. Python Dependencies
Install required packages:
```bash
G:/Sentinel/myenv/Scripts/python.exe -m pip install -r requirements_nlp.txt
```

## 🚀 Quick Start

### Option 1: Complete Pipeline Test
Run the automated test (generates posts for 30 seconds):
```bash
G:/Sentinel/myenv/Scripts/python.exe test_complete_pipeline.py
```

### Option 2: Manual Testing

#### Start Redis Consumer (Terminal 1)
```bash
G:/Sentinel/myenv/Scripts/python.exe enhanced_consumer.py --gemini-key AIzaSyDEEkyXceYLFknvkoVKYpIVnLWfSRz3wEY --firebase-cred sentinel-ed93e-firebase-adminsdk-fbsvc-996cb7c18e.json
```

#### Start Post Producer (Terminal 2)
```bash
G:/Sentinel/myenv/Scripts/python.exe simulator.py -r --duration 60
```

### Option 3: Individual Component Testing

#### Test NLP Engine Only
```bash
G:/Sentinel/myenv/Scripts/python.exe nlp_engine.py --gemini-key AIzaSyDEEkyXceYLFknvkoVKYpIVnLWfSRz3wEY --firebase-cred sentinel-ed93e-firebase-adminsdk-fbsvc-996cb7c18e.json --test
```

#### Test Basic Redis Consumer
```bash
G:/Sentinel/myenv/Scripts/python.exe redis_consumer.py
```

## 📊 What Each Component Does

### 1. **simulator.py** - Post Generator
- Generates realistic social media posts about disasters
- Supports multiple platforms (Twitter, Facebook, News, YouTube, Instagram)
- Streams posts to Redis in real-time
- Configurable timing and post types

### 2. **enhanced_consumer.py** - NLP Processing Pipeline
- Consumes posts from Redis streams
- Uses Gemini AI to verify genuine disasters vs false alarms
- Extracts structured information (locations, severity, impact)
- Stores verified disasters to Firebase
- Provides detailed statistics

### 3. **nlp_engine.py** - AI Processing Core
- Gemini API integration with few-shot prompting
- Disaster verification and information extraction
- Multi-language support with translation
- Firebase storage with structured data

### 4. **redis_consumer.py** - Basic Display
- Simple Redis consumer for viewing raw posts
- No AI processing - just displays stream data

## 🔧 System Architecture

```
[Social Media Simulator] 
    ↓ Redis Streams
[Enhanced Consumer with NLP]
    ↓ Gemini AI Verification
[Firebase Storage]
    ↓ Structured Data
[Analytics & Dashboards]
```

## 📈 Expected Output

### Genuine Disaster Detection
```json
{
  "processed": true,
  "genuine": true,
  "stored": true,
  "verification_result": {
    "is_genuine": true,
    "verification_score": 0.9,
    "reasoning": "Specific locations mentioned with urgent evacuation advice"
  },
  "extracted_data": {
    "disaster_type": "tsunami",
    "locations": [{"name": "Mumbai", "type": "city"}],
    "severity_indicators": ["evacuate immediately"],
    "impact_scale": "high"
  }
}
```

### False Alarm Detection
```json
{
  "processed": true,
  "genuine": false,
  "stored": false,
  "verification_result": {
    "is_genuine": false,
    "verification_score": 0.1,
    "reasoning": "Post mentions watching a movie, not a real event"
  }
}
```

## 🛠️ Troubleshooting

### Redis Connection Issues
- Ensure Docker is running
- Check if port 6379 is available
- Restart Redis: `docker restart redis-server`

### Gemini API Issues
- Verify API key is correct
- Check for quota limits
- Ensure internet connectivity

### Firebase Issues
- Verify credentials file path
- Check Firebase project permissions
- Ensure Firestore is enabled in Firebase Console

## 📋 Monitoring

The system provides real-time statistics:
- Total posts processed
- Verification accuracy rate
- Storage success rate
- Platform distribution
- Processing times

Run any component with verbose logging to see detailed operation information.