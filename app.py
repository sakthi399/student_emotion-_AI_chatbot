from flask import Flask, render_template, request, jsonify
import joblib
import random
import re

app = Flask(__name__)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

svm_model = joblib.load(os.path.join(BASE_DIR, "svm_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))


# Emotion responses
emotion_responses = {

"anxiety": [

"Exam anxiety is normal for many students. Try slowing your breathing and organizing your study time today. Study one subject for 45 minutes, take a 10-minute break, then revise key formulas or notes.",

"It sounds like exam anxiety is affecting your focus. Try starting with familiar topics first. Study for 40 minutes, rest briefly, then review important questions to gradually rebuild confidence.",

"Feeling anxious before exams can make studying difficult. Try a simple schedule today: morning for concept revision, afternoon for practice questions, and evening for reviewing mistakes and summaries.",

"Anxiety often increases when tasks feel overwhelming. Break your study into smaller parts. Focus on one chapter at a time and take short breaks to relax your mind and maintain steady concentration.",

"Exam worry is common during preparation. Begin with easier topics to build confidence, then move to harder ones. Studying step by step can help your mind feel calmer and more organized.",

"If anxiety feels strong right now, pause briefly and take deep breaths. Then create a short study list for today and complete topics one by one instead of thinking about everything at once.",

"Nervousness during exams is natural. Try revising important topics first, practice a few questions, and keep your study sessions short but focused to reduce pressure and improve understanding.",

"Anxiety often comes from worrying about results. Focus instead on today’s preparation. Plan two focused study sessions, review key concepts, and allow time for short breaks to keep your mind steady."

],

"stress": [

"Exam stress can build when there is too much to study. Try dividing subjects into smaller sections. Study one topic for 45 minutes, take a short break, and continue gradually.",

"You seem stressed about exams. A clear study routine can help reduce pressure. Morning: revise key concepts. Afternoon: solve practice questions. Evening: review mistakes and summarize important notes.",

"Academic stress is common during exam preparation. Focus on completing one chapter at a time. Small progress each day can gradually reduce pressure and improve confidence.",

"When stress feels heavy, slowing your pace may help. Study important topics first, take regular breaks, and avoid trying to memorize everything at once.",

"Exam stress sometimes appears when expectations feel high. Try setting small study goals for each session and reward yourself with short breaks after completing them.",

"Feeling mentally tired from studying is normal. Take a short walk or rest for a few minutes, then return to reviewing important notes or key concepts calmly.",

"Stress can affect focus and memory. Try practicing a few questions instead of only reading notes. Active practice often helps improve understanding and reduce pressure.",

"When study pressure increases, organization helps. Write down today’s topics, complete them one by one, and finish with a short review session to strengthen your preparation."

],

"positive": [

"It’s great to hear you’re feeling confident about your exams. Continue following your study routine, revise important concepts regularly, and practice past questions to maintain this positive momentum.",

"Your positive mindset can greatly support exam performance. Keep reviewing your notes, practice questions daily, and maintain a balanced routine with short breaks to keep your mind fresh.",

"Feeling motivated during exam preparation is a strong advantage. Continue studying consistently and focus on strengthening areas that need improvement while maintaining your confidence.",

"Confidence during exam preparation usually reflects good effort. Keep revising key concepts, practice problems regularly, and maintain steady progress in your study plan.",

"A positive attitude helps improve concentration and learning. Continue reviewing important topics and solving questions to strengthen your preparation and maintain this confidence.",

"It’s good to see your motivation during this exam period. Continue practicing important questions and revising concepts to maintain steady progress in your studies.",

"Staying positive during exam preparation can improve focus and memory. Keep following your study schedule and maintain regular revision to reinforce your understanding.",

"Your confidence suggests strong preparation habits. Continue revising your notes, practicing problems, and maintaining a balanced study routine to keep this positive mindset."

],

"neutral": [

"You seem calm right now, which is a good time for productive studying. Try reviewing important topics, organizing your notes, and planning your next study session carefully.",

"A balanced mindset can support effective exam preparation. Consider revising key chapters and practicing a few questions while your concentration is steady.",

"Remaining calm during exams helps improve focus. Use this time to review summaries, revise formulas, or clarify topics that need more understanding.",

"A neutral mindset can be helpful for learning. Try planning your study schedule for the day and focusing on important subjects first.",

"Being relaxed during preparation can improve productivity. Review important concepts and practice a few questions to strengthen your understanding.",

"You appear composed and balanced. This can be a good moment to revise key notes and organize topics that still require attention.",

"Staying calm helps improve concentration. Try revising core concepts or practicing questions step by step while your mind remains steady.",

"A steady emotional state supports learning. Use this time to review important chapters and plan your next study session effectively."

]

}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def predict_emotion(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    return svm_model.predict(vector)[0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json["message"]
    text = user_input.lower().strip()

    # Greeting
    if text in ["hi", "hello", "hey", "hai"]:
        return jsonify({
            "response": "Hello! I’m here to support your exam-related concerns. How are you feeling today?"
        })

    # Yes
    if text in ["yes", "yeah", "yep", "sure"]:
        return jsonify({
            "response": "I'm glad you're responding. Could you tell me more about how you're feeling regarding your exams?"
        })

    # No
    if text in ["no", "nope", "not really"]:
        return jsonify({
            "response": "That's okay. If you'd like, you can share what's bothering you."
        })
    # positive 
    if text in ["ok", "okay", "fine", "good", "alright" , "well"]:
       return jsonify({
            "response": "That's good to hear. If you'd like, you can tell me how your exam preparation is going, and I can suggest helpful study strategies."
        })

    # wish   
    if text in ["thankyou", "thanks", "thank you", "thx", "thank"]:
        return jsonify({
            "response": "It's my pleasure to help. If you have any more questions or need support, feel free to ask!"
        })
    #end 
    if text in ["bye", "goodbye", "see you", "see you later", "ok bye", "talk to you later"]:
        return jsonify({
            "response": "Goodbye! Take care and best of luck with your exams. Stay confident and keep preparing steadily."
   })

    # Gibberish detection
    if len(text.split()) == 1 and text not in vectorizer.vocabulary_:
        return jsonify({
            "response": "I’m not sure I understood that. Could you please rephrase your message?"
        })

    # Very short replies
    if len(text.split()) <= 2:
        return jsonify({
            "response": "Could you explain a bit more about how you're feeling?"
        })

    # ML prediction safely
    emotion = predict_emotion(user_input)

    
    response = random.choice(
        emotion_responses.get(emotion, ["I'm here to help."])
    )

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
