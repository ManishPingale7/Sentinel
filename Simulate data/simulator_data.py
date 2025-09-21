"""
Static data for the Social Media Disaster Simulator.
"""

# Hazard categories
HAZARD_CATEGORIES = [
    'Coastal_flooding', 'Cyclone', 'Rain', 'Tsunami', 'Erosion', 'Non'
]

# Supported languages
LANGUAGES = ['English', 'Hindi', 'Marathi', 'Tamil', 'Telugu', 'Konkani']

# Platforms
PLATFORMS = ['twitter', 'facebook', 'instagram', 'news', 'youtube']

# Post type distribution
POST_TYPE_WEIGHTS = {
    'hazard': 0.6,
    'false_alarm': 0.2,
    'noise': 0.2
}

# Location templates for different languages
LOCATIONS = {
    'English': [
        'Mumbai', 'Chennai', 'Kochi', 'Goa', 'Odisha coast', 'Andhra Pradesh',
        'Tamil Nadu', 'Kerala', 'Karnataka coast', 'Gujarat', 'West Bengal',
        'Visakhapatnam', 'Paradip', 'Mangalore', 'Calicut'
    ],
    'Hindi': [
        'मुंबई', 'चेन्नई', 'कोच्चि', 'गोवा', 'ओडिशा तट', 'आंध्र प्रदेश',
        'तमिलनाडु', 'केरल', 'कर्नाटक तट', 'गुजरात', 'पश्चिम बंगाल'
    ],
    'Marathi': [
        'मुंबई', 'कोकण', 'गोवा', 'रत्नागिरी', 'सिंधुदुर्ग', 'ठाणे',
        'पुणे', 'नागपूर', 'नाशिक', 'औरंगाबाद'
    ],
    'Tamil': [
        'சென்னை', 'கன்யாகுமாரி', 'रामेश्वरम्', 'तिरुनेल्वेली', 'कोयम्बतूर',
        'मदुरै', 'तिरुचिरापல्ली', 'सालेम', 'वेल्लोर'
    ],
    'Telugu': [
        'విశాఖపట్నం', 'విజయవాడ', 'హైదరాబాద్', 'గుంటూర్', 'నెల్లూరు',
        'కాకినాడ', 'రాజమండ్రి', 'తిరుపతి', 'కర్నూల్'
    ],
    'Konkani': [
        'गोवा', 'मंगळूर', 'उडुपी', 'कारवार', 'पणजी', 'मडगांव',
        'वास୍କो', 'मापुसा', 'बिचोलिम'
    ]
}

# Hazard-specific content templates
HAZARD_TEMPLATES = {
    'Coastal_flooding': {
        'English': [
            "🌊 Coastal flooding alert issued for {location}! Stay away from shore areas.",
            "High tide warning: {location} experiencing severe coastal flooding",
            "Emergency: Coastal areas of {location} under water. Evacuate immediately!",
            "Breaking: {location} faces major coastal flooding due to high tide",
            "⚠️ Coastal flood warning extended for {location} - avoid beach areas"
        ],
        'Hindi': [
            "🌊 {location} में तटीय बाढ़ की चेतावनी! समुद्री तटों से दूर रहें।",
            "उच्च ज्वार चेतावनी: {location} में गंभीर तटीय बाढ़",
            "आपातकाल: {location} के तटीय क्षेत्र पानी में डूबे। तुरंत निकलें!",
            "ब्रेकिंग: {location} में उच्च ज्वार के कारण बड़ी तटीय बाढ़"
        ],
        'Marathi': [
            "🌊 {location} साठी किनारी पुराची इशारा! समुद्राच्या किनाऱ्यापासून दूर राहा।",
            "भरती चेतावणी: {location} मध्ये गंभीर किनारी पूर",
            "आणीबाणी: {location} चे किनारी भाग पाण्याखाली। लगेच निघा!"
        ],
        'Tamil': [
            "🌊 {location} இல் கடலோர வெள்ளப்பெருக்கு எச்சரிக்கை! கடற்கரை பகுதிகளை விட்டு விலகுங்கள்.",
            "உயர் அலை எச்சரிக்கை: {location} இல் கடுமையான கடலோர வெள்ளம்",
            "அவசரம்: {location} கடலோர பகுதிகள் நீரில் மூழ்கின. உடனே வெளியேறுங்கள்!"
        ],
        'Telugu': [
            "🌊 {location} లో తీర వరద హెచ్చరిక! సముద్ర తీర ప్రాంతాలకు దూరంగా ఉండండి.",
            "అధిక అలల హెచ్చరిక: {location} లో తీవ్రమైన తీర వరదలు",
            "అత్యవసరం: {location} తీర ప్రాంతాలు నీటిలో మునిగాయి. వెంటనే వెళ్ళిపోండి!"
        ],
        'Konkani': [
            "🌊 {location} खातीर देगेर भरतीची ताकीद! दरयाच्या देगेर सावन लांब रावात।",
            "उंच भरती ताकीद: {location} मदीं गंभीर देगेर भरती",
            "घालमेळ: {location} च्या देगेर वाठार उदकांत बुडले। तत्काळ निघात!"
        ]
    },
    'Cyclone': {
        'English': [
            "🌀 Cyclone warning issued for {location}! Severe winds expected.",
            "Breaking: Cyclone approaching {location} - prepare for evacuation",
            "⚠️ Category 3 cyclone headed towards {location}. Take shelter now!",
            "Weather alert: {location} braces for severe cyclonic storm",
            "Emergency: Cyclone makes landfall near {location}"
        ],
        'Hindi': [
            "🌀 {location} के लिए चक्रवात चेतावनी! तेज हवाओं की उम्मीद।",
            "ब्रेकिंग: {location} की ओर चक्रवात - निकासी की तैयारी करें",
            "⚠️ श्रेणी 3 चक्रवात {location} की ओर बढ़ रहा है। अब आश्रय लें!",
            "मौसम अलर्ट: {location} गंभीर चक्रवाती तूफान के लिए तैयार"
        ],
        'Marathi': [
            "🌀 {location} साठी चक्रीवादळ चेतावणी! जोरदार वारे अपेक्षित।",
            "ब्रेकिंग: {location} कडे चक्रीवादळ येत आहे - स्थलांतराची तयारी करा",
            "⚠️ वर्ग 3 चक्रीवादळ {location} कडे जात आहे। आता आश्रय घ्या!"
        ],
        'Tamil': [
            "🌀 {location} இல் சூறாவளி எச்சரிக்கை! கடுமையான காற்று எதிர்பார்க்கப்படுகிறது.",
            "பிரேக்கிங்: {location} நோக்கி சூறாவளி நெருங்குகிறது - வெளியேற்றத்திற்கு தயாராகுங்கள்",
            "⚠️ வகை 3 சூறாவளி {location} நோக்கி செல்கிறது. இப்போதே பாதுகாப்பான இடத்திற்கு செல்லுங்கள்!"
        ],
        'Telugu': [
            "🌀 {location} కోసం తుఫాన్ హెచ్చరిక! తీవ్రమైన గాలులు ఊహించబడుతున్నాయి.",
            "బ్రేకింగ్: {location} వైపు తుఫాన్ చేరుకుంటోంది - తరలింపుకు సిద్ధపడండి",
            "⚠️ వర్గ 3 తుఫాన్ {location} వైపు వెళ్తోంది. ఇప్పుడే ఆశ్రయం తీసుకోండి!"
        ],
        'Konkani': [
            "🌀 {location} खातीर वावटळ ताकीद! जोरदार वारे अपेक्षित।",
            "ब्रेकिंग: {location} कडेन वावटळ येता - स्थलांतराची तयारी करात",
            "⚠️ वर्ग 3 वावटळ {location} कडेन वता। आतां आश्रय घेयात!"
        ]
    },
    'Rain': {
        'English': [
            "🌧️ Heavy rainfall warning for {location}! Flash floods possible.",
            "Weather alert: Torrential rains lash {location}",
            "⚠️ Orange alert: Heavy to very heavy rainfall in {location}",
            "Breaking: {location} receives record rainfall in 24 hours",
            "Monsoon update: Intense rainfall continues in {location}"
        ],
        'Hindi': [
            "🌧️ {location} के लिए भारी बारिश की चेतावनी! अचानक बाढ़ संभव।",
            "मौसम अलर्ट: {location} में मूसलाधार बारिश",
            "⚠️ नारंगी अलर्ट: {location} में भारी से अति भारी बारिश",
            "ब्रेकिंग: {location} में 24 घंटे में रिकॉर्ड बारिश"
        ],
        'Marathi': [
            "🌧️ {location} साठी मुसळधार पावसाची चेतावणी! अकस्मात पूर शक्य।",
            "हवामान अलर्ट: {location} मध्ये मुसळधार पाऊस",
            "⚠️ नारिंगी अलर्ट: {location} मध्ये मुसळधार ते अतिमुसळधार पाऊस"
        ],
        'Tamil': [
            "🌧️ {location} இல் கனமழை எச்சரிக்கை! திடீர் வெள்ளம் சாத்தியம்.",
            "வானிலை எச்சரிக்கை: {location} இல் பலத்த மழை",
            "⚠️ ஆரஞ்சு எச்சரிக்கை: {location} இல் கனமழையிலிருந்து மிகக் கனமழை"
        ],
        'Telugu': [
            "🌧️ {location} కోసం భారీ వర్షాల హెచ్చరిక! ఆకస్మిక వరదలు సాధ్యం.",
            "వాతావరణ హెచ్చరిక: {location} లో కుండపోత వర్షాలు",
            "⚠️ ఆరెంజ్ అలర్ట్: {location} లో భారీ నుండి అతి భారీ వర్షాలు"
        ],
        'Konkani': [
            "🌧️ {location} खातीर मुसळधार सावनाची ताकीद! अकस्मात पूर शक्य।",
            "हवामान अलर्ट: {location} मदीं मुसळधार सावन",
            "⚠️ केशरी अलर्ट: {location} मदीं मुसळधार ते अतिमुसळधार सावन"
        ]
    },
    'Tsunami': {
        'English': [
            "🌊 TSUNAMI WARNING for {location}! Move to higher ground immediately!",
            "EMERGENCY: Tsunami alert issued for {location} coastline",
            "⚠️ URGENT: Tsunami waves approaching {location} - evacuate now!",
            "Breaking: Earthquake triggers tsunami warning for {location}",
            "CRITICAL: All coastal residents of {location} must evacuate"
        ],
        'Hindi': [
            "🌊 {location} के लिए सुनामी चेतावनी! तुरंत ऊंची जगह जाएं!",
            "आपातकाल: {location} तटरेखा के लिए सुनामी अलर्ट",
            "⚠️ तत्काल: {location} की ओर सुनामी लहरें - अब निकलें!",
            "ब्रेकिंग: भूकंप से {location} के लिए सुनामी चेतावनी"
        ],
        'Marathi': [
            "🌊 {location} साठी त्सुनामी चेतावणी! तत्काळ उंच जागी जा!",
            "आणीबाणी: {location} किनारपट्टीसाठी त्सुनामी अलर्ट",
            "⚠️ तातडीने: {location} कडे त्सुनामी लाटा - आता निघा!"
        ],
        'Tamil': [
            "🌊 {location} இல் சுனாமி எச்சரிக்கை! உடனே உயரமான இடத்திற்கு செல்லுங்கள்!",
            "அவசரம்: {location} கடற்கரைக்கு சுனாமி எச்சரிக்கை",
            "⚠️ அவசரம்: {location} நோக்கி சுனாமி அலைகள் - இப்போதே வெளியேறுங்கள்!"
        ],
        'Telugu': [
            "🌊 {location} కోసం సునామీ హెచ్చరిక! వెంటనే ఎత్తైన ప్రాంతానికి వెళ్ళండి!",
            "అత్యవసరం: {location} తీరప్రాంతానికి సునామీ హెచ్చరిక",
            "⚠️ అత్యవసరం: {location} వైపు సునామీ అలలు - ఇప్పుడే వెళ్ళిపోండి!"
        ],
        'Konkani': [
            "🌊 {location} खातीर त्सुनामी ताकीद! तत्काळ उंच जागयेर वचात!",
            "घालमेळ: {location} देगेर पट्टी खातीर त्सुनामी अलर्ट",
            "⚠️ तत्काळ: {location} कडेन त्सुनामी लाटो - आतां निघात!"
        ]
    },
    'Erosion': {
        'English': [
            "🏖️ Coastal erosion alert: {location} beaches severely affected",
            "Environmental concern: Rapid erosion observed at {location}",
            "⚠️ Erosion warning: {location} coastline retreating rapidly",
            "Breaking: {location} faces severe coastal erosion threat",
            "Geological alert: Erosion damage reported in {location}"
        ],
        'Hindi': [
            "🏖️ तटीय कटाव अलर्ट: {location} के समुद्र तट गंभीर रूप से प्रभावित",
            "पर्यावरणीय चिंता: {location} में तेजी से कटाव देखा गया",
            "⚠️ कटाव चेतावनी: {location} तटरेखा तेजी से पीछे हट रही",
            "ब्रेकिंग: {location} को गंभीर तटीय कटाव का खतरा"
        ],
        'Marathi': [
            "🏖️ किनारी धूप अलर्ट: {location} चे समुद्रकिनारे गंभीरपणे प्रभावित",
            "पर्यावरणीय चिंता: {location} मध्ये जलद धूप दिसली",
            "⚠️ धूप चेतावणी: {location} किनारपट्टी वेगाने मागे सरकत आहे"
        ],
        'Tamil': [
            "🏖️ கடலோர அரிப்பு எச்சரிக்கை: {location} கடற்கரைகள் கடுமையாக பாதிக்கப்பட்டுள்ளன",
            "சுற்றுச்சூழல் கவலை: {location} இல் விரைவான அரிப்பு கண்டறியப்பட்டது",
            "⚠️ அரிப்பு எச்சரிக்கை: {location} கடற்கரை வேகமாக பின்வாங்குகிறது"
        ],
        'Telugu': [
            "🏖️ తీర కోత హెచ్చరిక: {location} తీరాలు తీవ్రంగా ప్రభావితమయ్యాయి",
            "పర్యావరణ ఆందోళన: {location} లో వేగవంతమైన కోత గమనించబడింది",
            "⚠️ కోత హెచ్చరిక: {location} తీరప్రాంతం వేగంగా వెనుకకు వెళ్తోంది"
        ],
        'Konkani': [
            "🏖️ देगेर धूप अलर्ट: {location} चे दरयाकिनारे गंभीरपणान प्रभावित",
            "पर्यावरणीय चिंता: {location} मदीं वेगळी धूप दिसली",
            "⚠️ धूप ताकीद: {location} देगेर पट्टी वेगान फाटीं सरकता"
        ]
    }
}

# False alarm templates
FALSE_ALARM_TEMPLATES = {
    'English': [
        "Urgent: Multiple reports of {hazard_type} in {location} area. Please be careful everyone",
        "Breaking: Just received news about {hazard_type} situation in {location}. Sharing for awareness",
        "My friend in {location} says {hazard_type} warnings are being issued. Stay safe!",
        "Forwarded: {hazard_type} alert for {location} - please share to help others",
        "Unconfirmed reports: {hazard_type} activity near {location}. Hope it's not serious"
    ],
    'Hindi': [
        "तत्काल: {location} क्षेत्र में {hazard_type} की कई रिपोर्ट्स। कृपया सभी सावधान रहें",
        "ब्रेकिंग: {location} में {hazard_type} स्थिति के बारे में अभी खबर मिली। जागरूकता के लिए साझा कर रहा हूं",
        "{location} में मेरे दोस्त का कहना है कि {hazard_type} चेतावनी जारी की जा रही है। सुरक्षित रहें!",
        "फॉरवर्ड: {location} के लिए {hazard_type} अलर्ट - कृपया दूसरों की मदद के लिए साझा करें"
    ],
    'Marathi': [
        "तातडीने: {location} भागात {hazard_type} च्या अनेक अहवाल। कृपया सर्वजण सावध राहा",
        "ब्रेकिंग: {location} मधील {hazard_type} परिस्थितीबद्दल आत्ताच बातमी मिळाली। जागरूकतेसाठी शेअर करत आहे",
        "{location} मधील माझ्या मित्राचे म्हणणे आहे की {hazard_type} चेतावणी दिली जात आहे। सुरक्षित राहा!"
    ],
    'Tamil': [
        "அவசரம்: {location} பகுதியில் {hazard_type} பற்றிய பல அறிக்கைகள். அனைவரும் கவனமாக இருங்கள்",
        "பிரேக்கிங்: {location} இல் {hazard_type} நிலைமை பற்றி இப்போது செய்தி கிடைத்தது। விழிப்புணர்வுக்காக பகிர்கிறேன்",
        "{location} இல் உள்ள என் நண்பர் {hazard_type} எச்சரிக்கைகள் வெளியிடப்படுவதாக கூறுகிறார். பாதுகாப்பாக இருங்கள்!"
    ],
    'Telugu': [
        "అత్యవసరం: {location} ప్రాంతంలో {hazard_type} గురించి అనేక నివేదికలు. దయచేసి అందరూ జాగ్రత్తగా ఉండండి",
        "బ్రేకింగ్: {location} లో {hazard_type} పరిస్థితి గురించి ఇప్పుడే వార్త వచ్చింది. అవగాహన కోసం షేర్ చేస్తున్నాను",
        "{location} లో నా స్నేహితుడు {hazard_type} హెచ్చరికలు జారీ చేయబడుతున్నాయని చెప్పారు. భద్రంగా ఉండండి!"
    ],
    'Konkani': [
        "तत्काळ: {location} वाठारांत {hazard_type} च्या कितल्याश अहवाल। उपकार करून सगळे सावधान रावात",
        "ब्रेकिंग: {location} मदल्या {hazard_type} परिस्थितीविशीं आतां बातमी मेळळी। जागरूकताखातीर शेअर करताम",
        "{location} मदल्या माझ्या इश्टाचो म्हण आसा कि {hazard_type} ताकीद दिवतात। सुरक्षित रावात!"
    ]
}

# YouTube video titles and descriptions
YOUTUBE_TEMPLATES = {
    'hazard': {
        'English': {
            'titles': [
                "🚨 LIVE: {hazard} hits {location} - Evacuation in progress",
                "BREAKING: Massive {hazard} devastates {location} | Latest Updates",
                "⚠️ {hazard} Emergency: {location} under threat - What you need to know",
                "URGENT: {hazard} warning for {location} - Stay safe everyone!",
                "{hazard} DISASTER: {location} faces unprecedented damage"
            ],
            'descriptions': [
                "Emergency situation developing in {location} as {hazard} continues to impact the region. Local authorities have issued evacuation orders for coastal areas. Stay tuned for live updates and safety information.",
                "Breaking news coverage of the {hazard} affecting {location}. We're bringing you the latest updates from ground zero. Please share this video to spread awareness and help others stay safe.",
                "Live coverage of {hazard} emergency in {location}. Disaster management teams are working around the clock. Follow official evacuation routes if you're in the affected area."
            ]
        },
        'Hindi': {
            'titles': [
                "🚨 लाइव: {location} में {hazard} की मार - निकासी जारी",
                "ब्रेकिंग: {location} में भारी {hazard} | नवीनतम अपडेट",
                "⚠️ {hazard} आपातकाल: {location} खतरे में - जानें क्या करें",
                "तत्काल: {location} के लिए {hazard} चेतावनी - सुरक्षित रहें!",
                "{hazard} आपदा: {location} अभूतपूर्व क्षति का सामना कर रहा है"
            ],
            'descriptions': [
                "{location} में {hazard} के कारण आपातकालीन स्थिति। स्थानीय अधिकारियों ने तटीय क्षेत्रों के लिए निकासी के आदेश जारी किए हैं। लाइव अपडेट और सुरक्षा जानकारी के लिए बने रहें।",
                "{location} को प्रभावित करने वाली {hazard} की ब्रेकिंग न्यूज कवरेज। हम आपके लिए ग्राउंड जीरो से नवीनतम अपडेट ला रहे हैं। कृपया इस वीडियो को साझा करें ताकि जागरूकता फैल सके और दूसरों की मदद की जा सके।",
                "{hazard} आपातकालीन स्थितीवर थेट कवरेज. आपत्ती व्यवस्थापन संघटन रात्रंदिवस काम करत आहेत. तुम्ही प्रभावित क्षेत्रात असाल तर अधिकृत निर्वासन मार्गांचे पालन करा."
            ]
        }
    },
    'false_alarm': {
        'English': "Debunking {hazard_category} Rumors in {location} - Fact Check",
        'Hindi': "{location} में {hazard_category} की अफवाहों का पर्दाफाश - फैक्ट चेक",
        'Marathi': "{location} मधील {hazard_category} अफवांचे खंडन - फॅक्ट चेक",
        'Tamil': "{location} இல் {hazard_category} வதந்திகளை உடைத்தல் - உண்மை சரிபார்ப்பு",
        'Telugu': "{location} లో {hazard_category} పుకార్లను తొలగించడం - ఫ్యాక్ట్ చెక్",
    },
    'noise': {
        'English': [
            "My reaction to weather forecast vs reality 😂",
            "When mom says there's a storm but you still go out",
            "POV: You're from {location} and see one cloud ☁️",
            "Rating disaster movies vs real life experience",
            "Why I don't trust weather apps anymore"
        ],
        'Hindi': [
            "मौसम पूर्वानुमान बनाम वास्तविकता पर मेरी प्रतिक्रिया 😂",
            "जब माँ कहती है कि तूफान है लेकिन तुम फिर भी बाहर जाते हो",
            "POV: आप {location} से हैं और एक बादल देखते हैं ☁️",
        ],
        'Marathi': [
            "हवामान अंदाज विरुद्ध वास्तविकता यावर माझी प्रतिक्रिया 😂",
            "जेव्हा आई म्हणते वादळ आहे पण तुम्ही तरीही बाहेर जाता",
            "POV: तुम्ही {location} चे आहात आणि एक ढग पाहता ☁️",
        ],
        'Tamil': [
            "வானிலை முன்னறிவிப்பு மற்றும் யதார்த்தத்திற்கு என் எதிர்வினை 😂",
            "புயல் என்று அம்மா சொல்லியும் வெளியே செல்லும்போது",
            "POV: நீங்கள் {location} ஐச் சேர்ந்தவர் மற்றும் ஒரு மேகத்தைப் பார்க்கிறீர்கள் ☁️",
        ],
        'Telugu': [
            "వాతావరణ సూచన వర్సెస్ వాస్తవికతపై నా స్పందన 😂",
            "అమ్మ తుఫాను ఉందని చెప్పినా మీరు బయటకు వెళ్ళినప్పుడు",
            "POV: మీరు {location} నుండి వచ్చారు మరియు ఒక మేఘాన్ని చూస్తారు ☁️",
        ],
    }
}

YOUTUBE_DESCRIPTIONS = {
    'hazard': {
        'English': "Emergency situation developing in {location} as {hazard_category} continues to impact the region. Local authorities have issued evacuation orders for coastal areas. Stay tuned for live updates and safety information.",
        'Hindi': "{location} में {hazard_category} के कारण आपातकालीन स्थिति। स्थानीय अधिकारियों ने तटीय क्षेत्रों के लिए निकासी के आदेश जारी किए हैं। लाइव अपडेट और सुरक्षा जानकारी के लिए बने रहें।",
        'Marathi': "{location} मध्ये {hazard_category} मुळे आपत्कालीन परिस्थिती. स्थानिक अधिकाऱ्यांनी किनारी भागांसाठी निर्वासन आदेश जारी केले आहेत. थेट अद्यतने आणि सुरक्षा माहितीसाठी संपर्कात रहा.",
        'Tamil': "{location} இல் {hazard_category} காரணமாக அவசரகால நிலைமை உருவாகியுள்ளது. கடலோரப் பகுதிகளுக்கு வெளியேற்ற உத்தரவுகளை உள்ளூர் அதிகாரிகள் வெளியிட்டுள்ளனர். நேரடி அறிவிப்புகள் மற்றும் பாதுகாப்பு தகவல்களுக்கு காத்திருங்கள்.",
        'Telugu': "{location} లో {hazard_category} కారణంగా అత్యవసర పరిస్థితి అభివృద్ధి చెందుతోంది. తీర ప్రాంతాలకు ఖాళీ చేయమని స్థానిక అధికారులు ఆదేశాలు జారీ చేశారు. ప్రత్యక్ష నవీకరణలు మరియు భద్రతా సమాచారం కోసం వేచి ఉండండి.",
    }
}

# YouTube comments
YOUTUBE_COMMENTS = {
    'English': [
        "Stay safe everyone! 🙏", "Thanks for the update", "Is this real?", 
        "My family is from there 😰", "Sending prayers", "First!",
        "When was this recorded?", "Fake news!", "Share this everywhere",
        "Government should do something", "Climate change is real",
        "Hope everyone is okay", "This looks scary", "Another disaster 😢"
    ],
    'Hindi': [
        "सभी सुरक्षित रहें! 🙏", "अपडेट के लिए धन्यवाद", "क्या यह सच है?",
        "मेरा परिवार वहां से है 😰", "प्रार्थनाएं भेज रहे हैं", "पहले!",
        "यह कब रिकॉर्ड किया गया?", "फेक न्यूज़!", "सबसे शेयर करें"
    ],
    'Marathi': [
        "सर्व सुरक्षित रहा! 🙏", "अपडेटबद्दल धन्यवाद", "हे खरे आहे का?",
        "माझे कुटुंब तिथले आहे 😰", "प्रार्थना पाठवत आहे", "पहिला!",
        "हे केव्हा रेकॉर्ड केले?", "खोटी बातमी!", "सर्वत्र शेअर करा"
    ],
    'Tamil': [
        "அனைவரும் பாதுகாப்பாக இருங்கள்! 🙏", "புதுப்பித்தலுக்கு நன்றி", "இது உண்மையா?",
        "என் குடும்பம் அங்கிருந்துதான் 😰", "பிரார்த்தனைகள் அனுப்புகிறேன்", "முதலில்!",
        "இது எப்போது பதிவு செய்யப்பட்டது?", "போலி செய்தி!", "எல்லா இடங்களிலும் பகிரவும்"
    ],
    'Telugu': [
        "అందరూ సురక్షితంగా ఉండండి! 🙏", "నవీకరణకు ధన్యవాదాలు", "ఇది నిజమేనా?",
        "నా కుటుంబం అక్కడిది 😰", "ప్రార్థనలు పంపుతున్నాను", "మొదటిది!",
        "ఇది ఎప్పుడు రికార్డ్ చేయబడింది?", "నకిలీ వార్తలు!", "అంతటా పంచుకోండి"
    ],
}

# News article templates
NEWS_TEMPLATES = {
    'hazard': {
        'English': {
            'headlines': [
                "{hazard} Strikes {location}: Thousands Evacuated as Emergency Declared",
                "Breaking: Severe {hazard} Hits {location}, Authorities Issue High Alert",
                "{location} Battles {hazard}: Rescue Operations Underway",
                "Emergency: {hazard} Devastates {location} Coastal Areas",
                "{hazard} Warning Extended for {location} as Conditions Worsen"
            ],
            'articles': [
                "{location}, {date} - A severe {hazard} has struck the coastal regions of {location}, prompting authorities to declare a state of emergency and order immediate evacuations of vulnerable areas.\n\nThe {hazard} began developing early this morning and has intensified rapidly, with meteorological departments issuing the highest level warnings for the region. Local disaster management authorities report that over {number} families have been moved to safety in evacuation centers.\n\n\"The situation is very serious and we are taking all necessary precautions,\" said {official_name}, District Collector of {location}. \"We urge all residents in coastal and low-lying areas to move to higher ground immediately.\"\n\nRescue teams from the National Disaster Response Force (NDRF) and state emergency services have been deployed to assist in evacuation efforts. Emergency shelters have been set up in schools and community centers across the district.\n\nThe India Meteorological Department (IMD) has forecast that the {hazard} will continue to impact the region for the next 24-48 hours. Fishermen have been advised not to venture into the sea, and all ports in the area have suspended operations.\n\nPower supply has been disrupted in several areas as a precautionary measure, and residents are advised to stay indoors and avoid unnecessary travel. Emergency helpline numbers have been activated for those needing assistance.\n\nThis is a developing story and will be updated as more information becomes available."
            ]
        },
        'Hindi': {
            'headlines': [
                "{location} में {hazard} का कहर: हजारों को सुरक्षित स्थानों पर पहुंचाया गया",
                "ब्रेकिंग: {location} में भीषण {hazard} , अधिकारियों ने जारी किया हाई अलर्ट",
                "{location} में {hazard} से मुकाबला: बचाव अभियान जारी",
                "आपातकाल: {location} के तटीय इलाकों में {hazard} का तांडव",
                "{hazard} चेतावनी बढ़ाई गई {location} के लिए क्योंकि स्थितियाँ बिगड़ती हैं"
            ],
            'articles': [
                "{location}, {date} - {location} के तटीय क्षेत्रों में एक गंभीर {hazard} ने दस्तक दी है, जिसके बाद अधिकारियों ने आपातकाल की स्थिति घोषित करते हुए संवेदनशील क्षेत्रों से तत्काल निकासी के आदेश दिए हैं।\n\n{hazard} आज सुबह विकसित होना शुरू हुआ और तेजी से तीव्र हो गया है। मौसम विभाग ने इस क्षेत्र के लिए उच्चतम स्तर की चेतावनी जारी की है। स्थानीय आपदा प्रबंधन अधिकारियों की रिपोर्ट के अनुसार {number} से अधिक परिवारों को निकालकर सुरक्षित स्थानों पर पहुंचाया गया है।\n\n\"{location} के जिला कलेक्टर {official_name} ने कहा, \"स्थिति बहुत गंभीर है और हम सभी आवश्यक सावधानियां बरत रहे हैं। हम तटीय और निचले इलाकों के सभी निवासियों से आग्रह करते हैं कि वे तुरंत ऊंचे स्थानों पर चले जाएं।\""
            ]
        }
    }
}

# Noise/joke templates
NOISE_TEMPLATES = {
    'English': [
        "Just saw my boss's face... definitely a natural disaster! 😂 #MondayMotivation",
        "My cooking caused more evacuation than any cyclone 🤣 #KitchenDisaster",
        "Rain forecast: 100% chance my ex will text me today ☔ #WeatherUpdate",
        "Breaking: Local man discovers tsunami in his bathroom after toilet overflow 🌊",
        "Tsunami warning: My mom found my browser history 😱 #FamilyDrama",
        "Cyclone update: Still spinning from last night's party 🌀 #Hangover",
        "Flood alert: My bank account after shopping 💸 #BrokeLife",
        "Emergency evacuation: Neighbor started karaoke night 🎤 #SaveYourselves",
        "Weather warning: My mood swings are more unpredictable than monsoons ⛈️",
        "Breaking news: Pizza delivery guy is the real hero during any disaster 🍕"
    ],
    'Hindi': [
        "अभी अपने बॉस का चेहरा देखा... निश्चित रूप से प्राकृतिक आपदा! 😂 #MondayMotivation",
        "मेरे खाना बनाने से किसी भी चक्रवात से ज्यादा निकासी हुई 🤣 #KitchenDisaster",
        "बारिश का पूर्वानुमान: आज मेरी एक्स के मैसेज आने की 100% संभावना ☔ #WeatherUpdate",
        "ब्रेकिंग: स्थानीय व्यक्ति ने टॉयलेट ओवरफ्लो के बाद बाथरूम में सुनामी खोजी 🌊"
    ],
    'Marathi': [
        "नुकताच माझ्या बॉसचा चेहरा पाहिला... नक्कीच नैसर्गिक आपत्ती! 😂 #MondayMotivation",
        "माझ्या स्वयंपाकामुळे कोणत्याही चक्रीवादळापेक्षा जास्त निकासी झाली 🤣 #KitchenDisaster",
        "पावसाचा अंदाज: आज माझ्या एक्सचा मेसेज येण्याची 100% शक्यता ☔ #WeatherUpdate"
    ],
    'Tamil': [
        "இப்போதுதான் என் பாஸின் முகத்தைப் பார்த்தேன்... நிச்சயமாக இயற்கை பேரழிவு! 😂 #MondayMotivation",
        "என் சமையலால் எந்த சூறாவளியையும் விட அதிக வெளியேற்றம் நடந்தது 🤣 #KitchenDisaster",
        "மழை முன்னறிவிப்பு: இன்று என் முன்னாள் காதலர் எனக்கு குறுந்தகவல் அனுப்ப 100% வாய்ப்பு ☔ #WeatherUpdate"
    ],
    'Telugu': [
        "ఇప్పుడే నా బాస్ ముఖం చూశాను... ఖచ్చితంగా ప్రకృతి విపత్తు! 😂 #MondayMotivation",
        "నా వంటవల్ల ఏ తుఫాన్ కంటే ఎక్కువ వెలివేత జరిగింది 🤣 #KitchenDisaster",
        "వర్షం అంచనా: ఈరోజు నా ఎక్స్ నాకు మెసేజ్ పంపే 100% అవకాశం ☔ #WeatherUpdate"
    ],
    'Konkani': [
        "आतां माझ्या बॉसाचो चेहरो पळयलो... निश्चितपणान नैसर्गिक आपत्ती! 😂 #MondayMotivation",
        "माझ्या स्वयंपाकान कोणत्याय वावटळापरस चड निकास जाली 🤣 #KitchenDisaster",
        "सावनाचो अंदाज: आज माझ्या एक्साचो मेसेज येवचो 100% संभव ☔ #WeatherUpdate"
    ]
}

YOUTUBE_CHANNEL_PATTERNS = {
    'English': ['News Today', 'Live Reports', 'Global Watch', 'India Now', 'Weather Channel'],
    'Hindi': ['आज की खबर', 'लाइव रिपोर्ट', 'भारत समाचार', 'मौसम समाचार'],
    'Marathi': ['आजची बातमी', 'थेट प्रक्षेपण', 'महाराष्ट्र वार्ता'],
    'Tamil': ['இன்றைய செய்திகள்', 'நேரடி அறிக்கை', 'உலக செய்திகள்'],
    'Telugu': ['ఈరోజు వార్తలు', 'ప్రత్యక్ష ప్రసారం', 'ప్రపంచ వార్తలు'],
}

YOUTUBE_TITLES = {
    'hazard': {
        'English': [
            "🚨 LIVE: {hazard} hits {location} - Evacuation in progress",
            "BREAKING: Massive {hazard} devastates {location} | Latest Updates",
        ],
        'Hindi': [
            "🚨 लाइव: {location} में {hazard} की मार - निकासी जारी",
            "ब्रेकिंग: {location} में भारी {hazard} | नवीनतम अपडेट",
        ]
    },
    'false_alarm': {
        'English': "Debunking {hazard_category} Rumors in {location} - Fact Check",
        'Hindi': "{location} में {hazard_category} की अफवाहों का पर्दाफाश - फैक्ट चेक",
    },
    'noise': {
        'English': [
            "My reaction to weather forecast vs reality 😂",
            "When mom says there's a storm but you still go out",
        ],
        'Hindi': [
            "मौसम पूर्वानुमान बनाम वास्तविकता पर मेरी प्रतिक्रिया 😂",
            "जब माँ कहती है कि तूफान है लेकिन तुम फिर भी बाहर जाते हो",
        ],
    }
}

YOUTUBE_COMMENTS_TEMPLATES = YOUTUBE_COMMENTS

# Twitter short templates
TWITTER_SHORT_TEMPLATES = {
    'English': [
        "🚨 {hazard} alert for {location}! Stay safe everyone #DisasterAlert",
        "Breaking: {hazard} hits {location}. Thoughts and prayers 🙏 #Emergency",
        "⚠️ {hazard} warning issued for {location}. Avoid coastal areas! #Safety",
        "Just heard about {hazard} in {location}. Hope everyone is okay 😟",
        "{hazard} update: {location} under emergency. Follow official advisories #StaySafe",
        # Casual/slang additions
        "Bruh the {hazard} situation in {location} is getting real bad 😰 #StaySafe",
        "OMG {location} hit by {hazard} again!! When will this stop?? 😭",
        "Guys be careful! {hazard} warning for {location} is no joke 💯 #Emergency",
        "Fam please stay indoors! {hazard} in {location} looks scary af 😨",
        "NGL this {hazard} thing in {location} has me shook... prayers up 🙏✨",
        "Broski check the news! {hazard} alert for {location} rn 📺⚠️",
        "Yo {location} people stay safe!! This {hazard} looks mental 🌊💨"
    ],
    'Hindi': [
        "🚨 {location} में {hazard} की चेतावनी! सभी सुरक्षित रहें #DisasterAlert",
        "ब्रेकिंग: {location} में {hazard}। प्रार्थनाएं 🙏 #Emergency", 
        "⚠️ {location} के लिए {hazard} चेतावनी। तटीय क्षेत्रों से बचें! #Safety",
        # Casual/slang additions
        "यार {location} में {hazard} का हाल देखो!! बहुत डरावना लग रहा है 😰",
        "अरे भाई {location} में फिर से {hazard}! कब तक ऐसे चलेगा? 😭",
        "दोस्तों घर में रहो! {location} में {hazard} कोई मजाक नहीं है 💯",
        "भाई लोग सावधान रहो! {location} में {hazard} बहुत खतरनाक दिख रहा है 😨"
    ],
    'Marathi': [
        "अरे {location} मध्ये {hazard} पाहा! खूप भयानक आहे 😰",
        "मित्रांनो सावध रहा! {location} मध्ये {hazard} धोकादायक आहे 💯",
        "अगं {location} च्या लोकांनो घरात रहा! {hazard} कोणता मजाक नाही 😨"
    ]
}

# Instagram/Facebook casual templates
SOCIAL_CASUAL_TEMPLATES = {
    'English': [
        "Guys the weather here in {location} is getting crazy 😰 {hazard} warning everywhere... staying indoors today! #WeatherUpdate #StaySafe",
        "Mom just called panicking about {hazard} news in {location} 📱 Anyone else getting these warnings? #Family #Weather",
        "Not gonna lie, this {hazard} stuff in {location} has me worried... sending good vibes to everyone there 💙 #SendingLove",
        "Weather app going crazy with {hazard} alerts for {location} 📱⚠️ Time to stock up on snacks and charge devices! #WeatherPrep",
        # Enhanced casual/slang versions
        "Bruhhh {location} weather is absolutely mental rn!! 😱 {hazard} alerts non-stop... time to binge Netflix indoors 📺🍿 #StayInside",
        "Fam the {hazard} situation in {location} is legit scary... sending all my love and prayers 💕🙏 #StaySafe #Prayers",
        "NGL kinda freaking out about this {hazard} thing in {location} 😨 Anyone else's family calling every 5 mins? 📞😅 #FamilyWorries",
        "Okay but like... this {hazard} alert for {location} came outta nowhere!! Weather apps be wildin' 📱⛈️ #WeatherStruck",
        "Me: *sees {hazard} warning for {location}* Also me: *immediately calls mom* 📞👵 Anyone else? 😂 #MomWorries #SafetyFirst",
        "POV: You're from {location} and your phone won't stop buzzing with {hazard} alerts 📱💥 RIP my battery 🔋😭 #EmergencyMode",
        "That feeling when {location} gets hit with {hazard} and your entire family group chat explodes 💬💥 #FamilyDrama #StaySafe"
    ],
    'Hindi': [
        "यार {location} में मौसम बहुत खराब हो रहा है 😰 हर जगह {hazard} की चेतावनी... आज घर में ही रहूंगा! #WeatherUpdate #StaySafe",
        "माम ने अभी फोन करके {location} में {hazard} की खबर सुनाकर परेशान किया 📱 और किसी को भी ये चेतावनी मिली है? #Family #Weather",
        # Enhanced casual versions
        "अरे भाई {location} का मौसम एकदम पागल हो गया है!! 😱 {hazard} के अलर्ट आते ही रह रहे हैं... घर में बैठकर सीरियल देखना पड़ेगा 📺",
        "यार ये {hazard} वाली बात {location} में सच में डरावनी है... सबके लिए प्रार्थना कर रहा हूं 💕🙏 #StaySafe",
        "सच कहूं तो {location} में {hazard} की वजह से घबराहट हो रही है 😨 तुम्हारे घर वाले भी हर 5 मिनट में फोन कर रहे हैं क्या? 📞😅",
        "भाई {location} के लिए {hazard} का अलर्ट देखकर मम्मी को तुरंत फोन किया 📞👵 और कोई ऐसा करता है? 😂 #MomCare",
        "जब {location} में {hazard} आता है और पूरा family WhatsApp group हिल जाता है 💬💥 #FamilyTension #StaySafe"
    ],
    'Marathi': [
        "अरे {location} मधला weather एकदम crazy झाला आहे!! 😱 {hazard} चे alerts येतच राहतात... आज घरातच बसायचं 📺",
        "ही {hazard} ची गोष्ट {location} मध्ये खरंच scary आहे यार... सगळ्यांसाठी प्रार्थना करतोय 💕🙏",
        "खरं सांगायचं तर {location} मधल्या {hazard} मुळे भीती वाटतेय 😨 तुमच्या घरच्यांनी पण फोन केला का? 📞😅"
    ]
}

# Code-switching templates
CODE_SWITCHING_TEMPLATES = {
    'Hindi-English': [
        "Mumbai mein {hazard} ho gaya 😡 trains bandh hai #MumbaiFloods #StaySafe",
        "Yaar this {hazard} in {location} is so scary!! सब लोग safe रहो please 🙏",
        "Breaking: {location} mein massive {hazard}!! Everyone please evacuate अभी!! 🚨",
        "OMG {location} का {hazard} देखकर दिल दहल गया 😰 prayers for all families there",
        "Guys {location} mein {hazard} situation बहुत serious है... avoid going out! #EmergencyAlert",
        "Just saw {location} {hazard} videos... ye तो बहुत ही scary है यार 😨📱",
        "Family in {location} saying {hazard} situation गंभीर है... everyone stay safe! 💙",
        "Weather app shows {hazard} warning for {location}... सब घर पर रहो guys! ⚠️🏠"
    ],
    'Marathi-English': [
        "Pune madhe {hazard} आला आहे!! सगळे careful रहा guys 😰 #PuneRains",
        "This {hazard} in {location} खरंच scary आहे यार... prayers for everyone 🙏",
        "Breaking news: {location} मध्ये massive {hazard}! सगळ्यांनी evacuate करा अभी! 🚨",
        "OMG {location} च्या {hazard} ची photos पाहून काळजं ठेंगणं झालं 😨",
        "Guys {location} मधला {hazard} situation खूप serious आहे... बाहेर जाऊ नका! #Emergency"
    ],
    'Tamil-English': [
        "{location} la {hazard} வந்துட்டு!! Everyone please safe ஆ இருங்க 😰",
        "Breaking: {location} mein serious {hazard}... எல்லாரும் careful ஆ இருங்க! 🚨",
        "This {hazard} situation in {location} romba scary ஆ இருக்கு... prayers 🙏",
        "Weather alert: {location} ku {hazard} warning... घर मे रहो guys! ⚠️"
    ],
    'Telugu-English': [
        "{location} lo {hazard} వచ్చింది!! Everyone please safe గా ఉండండి 😰",
        "Breaking news: {location} లో massive {hazard}... అందరూ careful ఉండండి! 🚨",
        "This {hazard} in {location} చాలా scary ఉంది yaar... prayers for all 🙏"
    ]
}

# Author names
AUTHOR_NAMES = {
    'English': [
        'Rajesh Kumar', 'Priya Sharma', 'Amit Singh', 'Neha Patel', 'Arjun Reddy',
        'Kavya Nair', 'Rohit Verma', 'Ananya Iyer', 'Vikram Chopra', 'Meera Joshi'
    ],
    'Hindi': [
        'राजेश कुमार', 'प्रिया शर्मा', 'अमित सिंह', 'नेहा पटेल', 'अर्जुन रेड्डी',
        'काव्या नायर', 'रोहित वर्मा', 'अनन्या अय्यर', 'विक्रम चोपड़ा', 'मीरा जोशी'
    ],
    'Marathi': [
        'राजेश कुमार', 'प्रिया शर्मा', 'अमित सिंह', 'नेहा पटेल', 'अर्जुन रेड्डी'
    ],
    'Tamil': [
        'ராஜேஷ் குமார்', 'பிரியா ஷர்மா', 'அமித் சிங்', 'நேஹா படேல்', 'அர்ஜுன் ரெட்டி'
    ],
    'Telugu': [
        'రాజేష్ కుమార్', 'ప్రియా శర్మ', 'అమిత్ సింగ్', 'నేహా పటేల్', 'అర్జున్ రెడ్డి'
    ],
    'Konkani': [
        'राजेश कुमार', 'प्रिया शर्मा', 'अमित सिंह', 'नेहा पटेल', 'अर्जुन रेड्डी'
    ]
}

# News organizations
NEWS_ORGS = [
    'Coastal Times', 'Regional News Network', 'Weather Watch India', 
    'Disaster Alert News', 'Local Herald', 'Emergency Broadcasting',
    'समाचार नेटवर्क', 'क्षेत्रीय समाचार', 'मौसम अलर्ट न्यूज'
]

# Hashtag templates
HASHTAGS = {
    'Coastal_flooding': ['#CoastalFlood', '#FloodAlert', '#HighTide', '#CoastalWarning', '#FloodSafety'],
    'Cyclone': ['#CycloneAlert', '#StormWarning', '#CycloneWatch', '#WeatherAlert', '#DisasterPrep'],
    'Rain': ['#HeavyRain', '#MonsoonAlert', '#RainWarning', '#FloodRisk', '#WeatherUpdate'],
    'Tsunami': ['#TsunamiAlert', '#TsunamiWarning', '#EmergencyEvacuation', '#TsunamiWatch', '#DisasterAlert'],
    'Erosion': ['#CoastalErosion', '#Erosion', '#EnvironmentalAlert', '#ClimateChange', '#CoastalManagement'],
    'Non': ['#Funny', '#Meme', '#Joke', '#Life', '#Random', '#Comedy', '#Humor', '#Daily']
}

# Emojis for different hazards
EMOJIS = {
    'Coastal_flooding': ['🌊', '⚠️', '🚨', '🏃‍♂️', '🏠'],
    'Cyclone': ['🌀', '💨', '⚠️', '🚨', '🏠'],
    'Rain': ['🌧️', '☔', '⛈️', '💧', '⚠️'],
    'Tsunami': ['🌊', '🚨', '⚠️', '🏃‍♂️', '🆘'],
    'Erosion': ['🏖️', '⚠️', '🌊', '🔍', '📉'],
    'Non': ['😂', '🤣', '😎', '🔥', '💯', '🎉', '✨']
}

USER_NAMES_BY_LANGUAGE = {
    'English': {
        'first': ['Rohan', 'Priya', 'Amit', 'Sneha', 'Vikram'],
        'last': ['Kumar', 'Sharma', 'Patel', 'Singh', 'Gupta']
    },
    'Hindi': {
        'first': ['रोहन', 'प्रिया', 'अमित', 'स्नेहा', 'विक्रम'],
        'last': ['कुमार', 'शर्मा', 'पटेल', 'सिंह', 'गुप्ता']
    },
    'Marathi': {
        'first': ['रोहन', 'प्रिया', 'अमित', 'स्नेहा', 'विक्रम'],
        'last': ['कुलकर्णी', 'देशपांडे', 'जोशी', 'पाटील', 'चव्हाण']
    },
    'Tamil': {
        'first': ['ரோஹன்', 'பிரியா', 'அமித்', 'சினேகா', 'விக்ரம்'],
        'last': ['குமார்', 'சர்மா', 'படேல்', 'சிங்', 'குப்தா']
    },
    'Telugu': {
        'first': ['రోహన్', 'ప్రియ', 'అమిత్', 'స్నేహ', 'విక్రమ్'],
        'last': ['కుమార్', 'శర్మ', 'పటేల్', 'సింగ్', 'గుప్తా']
    }
}

USERNAME_PATTERNS = [
    '{first}{last}{year}',
    '{first}_{last}',
    '{last}{first}{num}',
    '{city}_{first}',
]

BIRTH_YEARS = ['88', '92', '95', '01', '03']
CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']

BIO_TEMPLATES = {
    'news': {
        'English': ["Official account for Breaking News 24/7. Follow us for the latest updates.", "Your trusted source for news and analysis."],
        'Hindi': ["ब्रेकिंग न्यूज़ 24/7 का आधिकारिक अकाउंट। नवीनतम अपडेट के लिए हमें फॉलो करें।", "समाचार और विश्लेषण के लिए आपका विश्वसनीय स्रोत।"]
    },
    'government': {
        'English': ["Official account of the Disaster Management Authority.", "Updates from the Ministry of Home Affairs."],
        'Hindi': ["आपदा प्रबंधन प्राधिकरण का आधिकारिक अकाउंट।", "गृह मंत्रालय से अपडेट।"]
    },
    'journalist': {
        'English': ["Reporter @ NewsHub India. Views are my own.", "Investigative journalist covering climate and weather."],
        'Hindi': ["रिपोर्टर @ न्यूज़हब इंडिया। विचार मेरे अपने हैं।", "जलवायु और मौसम को कवर करने वाले खोजी पत्रकार।"]
    },
    'organization': {
        'English': ["NGO working on climate resilience and disaster relief.", "Community-based organization for coastal safety."],
        'Hindi': ["जलवायु लचीलापन और आपदा राहत पर काम कर रहे एनजीओ।", "तटीय सुरक्षा के लिए समुदाय-आधारित संगठन।"]
    },
    'personal': {
        'English': ["Just a regular person.", "Love dogs, travel, and food.", "Engineer | Father | Husband"],
        'Hindi': ["बस एक आम इंसान।", "कुत्ते, यात्रा और भोजन से प्यार है।", "इंजीनियर | पिता | पति"]
    }
}

BASE_ENGAGEMENT_RATES = {
    'twitter': {'likes': 0.01, 'retweets': 0.005, 'replies': 0.001},
    'facebook': {'likes': 0.02, 'shares': 0.003, 'comments': 0.002},
    'instagram': {'likes': 0.03, 'comments': 0.0015, 'saves': 0.001},
    'youtube': {'likes': 0.025, 'dislikes': 0.002, 'comments': 0.005}
}

TYPE_MULTIPLIERS = {
    'hazard': 1.5,
    'false_alarm': 1.2,
    'noise': 0.8
}

ACCOUNT_MULTIPLIERS = {
    'official': 2.0,
    'personal': 1.0
}

PLATFORM_HASHTAG_COUNTS = {
    'twitter': (1, 4),
    'instagram': (5, 15),
    'facebook': (1, 5),
    'youtube': (3, 8),
    'news': (2, 6)
}

INSTAGRAM_EXTRA_HASHTAGS = ['#InstaGood', '#PhotoOfTheDay', '#InstaDaily', '#ExplorePage']
TWITTER_EXTRA_HASHTAGS = ['#Trending', '#Viral', '#BreakingNews']
YOUTUBE_EXTRA_HASHTAGS = ['#YouTube', '#Video', '#Creator']

NEWS_HEADLINE_TEMPLATES = {
    'hazard': {
        'English': ["{hazard_category} Strikes {location}: Thousands Evacuated", "Severe {hazard_category} Hits {location}, High Alert Issued"],
        'Hindi': ["{location} में {hazard_category} का कहर: हजारों को निकाला गया", "{location} में गंभीर {hazard_category}, हाई अलर्ट जारी"]
    },
    'false_alarm': {
        'English': ["Officials Debunk {hazard_category} Rumors in {location}", "No Threat of {hazard_category} in {location}, Authorities Confirm"],
        'Hindi': ["अधिकारियों ने {location} में {hazard_category} की अफवाहों का खंडन किया", "{location} में {hazard_category} का कोई खतरा नहीं, अधिकारियों ने पुष्टि की"]
    },
    'noise': {
        'English': ["Local Resident's Humorous Post on Weather Goes Viral", "Light Moments Amidst Serious Warnings in {location}"],
        'Hindi': ["मौसम पर स्थानीय निवासी का मजाकिया पोस्ट वायरल हुआ", "{location} में गंभीर चेतावनियों के बीच हल्के पल"]
    }
}

NEWS_ARTICLE_CONTENT = {
    'hazard': {
        'English': "A severe {hazard_category} has struck the coastal regions of {location}, prompting authorities to declare a state of emergency. Rescue operations are underway.",
        'Hindi': "{location} के तटीय क्षेत्रों में एक गंभीर {hazard_category} ने दस्तक दी है, जिसके बाद अधिकारियों ने आपातकाल की स्थिति घोषित कर दी है। बचाव अभियान जारी है।"
    }
}

NEWS_OUTLETS = {
    'English': ['India Today', 'NDTV', 'The Times of India', 'Hindustan Times'],
    'Hindi': ['आज तक', 'एनडीटीवी इंडिया', 'दैनिक भास्कर', 'अमर उजाला']
}

NEWS_AUTHORS = {
    'English': ['Priya Singh', 'Rohan Gupta', 'Amit Sharma', 'Sneha Patel'],
    'Hindi': ['प्रिया सिंह', 'रोहन गुप्ता', 'अमित शर्मा', 'स्नेहा पटेल']
}

INSTAGRAM_CAPTIONS = {
    'hazard': {
        'English': "Staying safe indoors as {hazard_category} hits {location}. Hope everyone is okay!",
        'Hindi': "{location} में {hazard_category} के चलते घर के अंदर सुरक्षित। उम्मीद है सब ठीक होंगे!"
    },
    'false_alarm': {
        'English': "That {hazard_category} rumor about {location} was fake news! Always verify before you share.",
        'Hindi': "{location} के बारे में वह {hazard_category} की अफवाह झूठी खबर थी! शेयर करने से पहले हमेशा सत्यापित करें।"
    },
    'noise': {
        'English': ["Just another beautiful day in {location}!", "Enjoying the little things."],
        'Hindi': ["{location} में एक और खूबसूरत दिन!", "छोटी-छोटी चीजों का आनंद ले रहा हूं।"]
    }
}

FACEBOOK_POSTS = {
    'hazard': {
        'English': "Update from {location}: The {hazard_category} situation is serious. Please follow all safety advisories from local authorities. Our thoughts are with everyone affected.",
        'Hindi': "{location} से अपडेट: {hazard_category} की स्थिति गंभीर है। कृपया स्थानीय अधिकारियों के सभी सुरक्षा सलाह का पालन करें। हमारी संवेदनाएं प्रभावित सभी लोगों के साथ हैं।"
    },
    'false_alarm': {
        'English': "Important clarification: The news about a {hazard_category} in {location} is NOT true. Please do not spread unverified information. Stay safe and rely on official sources.",
        'Hindi': "महत्वपूर्ण स्पष्टीकरण: {location} में {hazard_category} के बारे में खबर सच नहीं है। कृपया असत्यापित जानकारी न फैलाएं। सुरक्षित रहें और आधिकारिक स्रोतों पर भरोसा करें।"
    },
    'noise': {
        'English': ["What a beautiful sunset in {location} today!", "Feeling blessed and grateful."],
        'Hindi': ["आज {location} में कितना सुंदर सूर्यास्त है!", "धन्य और आभारी महसूस कर रहा हूं।"]
    }
}
