# Emotion Detection from Social Media Text

def detect_emotion(text):
    text = text.lower()

    emotions = {
        "Happy": [
            "happy", "joy", "excited", "love", "great",
            "awesome", "wonderful", "amazing", "good"
        ],

        "Sad": [
            "sad", "unhappy", "cry", "crying", "lonely",
            "hurt", "bad", "miss", "depressed"
        ],

        "Angry": [
            "angry", "hate", "mad", "furious", "annoyed",
            "irritated", "rage"
        ],

        "Fear": [
            "afraid", "fear", "scared", "danger",
            "worried", "nervous", "terrified"
        ],

        "Surprise": [
            "surprised", "wow", "unexpected",
            "shocked", "suddenly"
        ],

        "Love": [
            "love", "lovely", "care", "dear", "beautiful"
        ]
    }

    emotion_scores = {}

    for emotion, keywords in emotions.items():
        score = 0

        for word in keywords:
            if word in text:
                score += 1

        emotion_scores[emotion] = score

    detected_emotion = max(
        emotion_scores,
        key=emotion_scores.get
    )

    highest_score = emotion_scores[detected_emotion]

    if highest_score == 0:
        return "Neutral", 0

    total_score = sum(emotion_scores.values())

    confidence = (highest_score / total_score) * 100

    return detected_emotion, round(confidence, 2)