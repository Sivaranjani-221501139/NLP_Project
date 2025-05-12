from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import PyPDF2
from keybert import KeyBERT
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  

logging.basicConfig(level=logging.DEBUG)

llama_model_tutor = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3b-hf")  
llama_tokenizer_tutor = AutoTokenizer.from_pretrained("meta-llama/Llama-3b-hf")


bart_model_quiz_exam = AutoModelForCausalLM.from_pretrained("facebook/bart-large")  
bart_tokenizer_quiz_exam = AutoTokenizer.from_pretrained("facebook/bart-large")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/tutor')
def tutor():
    return render_template('tutor.html')

@app.route('/exam')
def exam():
    return render_template('exam.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')


@app.route('/ask-tutor', methods=['POST'])
def ask_tutor():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"answer": "⚠️ Please ask a valid question."})

    try:
        answer = ask_llama(question)
    except Exception as e:
        return jsonify({"answer": "⚠️ Sorry, something went wrong with the AI response."})

    return jsonify({"answer": answer})

def ask_llama(question):
    """Fetch response from LLaMA model for tutor."""
    inputs = llama_tokenizer_tutor(question, return_tensors="pt")
    outputs = llama_model_tutor.generate(inputs['input_ids'], max_length=650, num_return_sequences=1)
    answer = llama_tokenizer_tutor.decode(outputs[0], skip_special_tokens=True)
    return answer


def extract_pdf_text(file):
    """Extract text content from the uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_topic_content(pdf_text, topic):
    """Find relevant text for the topic."""
    lines = pdf_text.split("\n")
    filtered_content = [line for line in lines if topic.lower() in line.lower()]
    return " ".join(filtered_content).strip()

def extract_keywords(content, num_keywords=20):
    """Extract keywords from content using KeyBERT."""
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(content, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=num_keywords)
    return [kw[0] for kw in keywords]


def generate_mcqs_bart(topic, keywords, content, num_questions, difficulty):
    prompt = f"""
You are an expert quiz generator. Generate {num_questions} {difficulty} level MCQs based on the study content below.

--- Study Content ---
{content}

--- Keywords ---
{', '.join(keywords)}

--- Format ---
Q: <question>
A. <option1>
B. <option2>
C. <option3>
D. <option4>
"""
    inputs = bart_tokenizer_quiz_exam(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = bart_model_quiz_exam.generate(inputs['input_ids'], max_length=250, num_return_sequences=1)
    quiz_output = bart_tokenizer_quiz_exam.decode(outputs[0], skip_special_tokens=True)
    return quiz_output



@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    if 'file' not in request.files:
        return jsonify({"error": "❌ No file part"}), 400
    
    file = request.files['file']
    topic = request.form.get('topic', "").strip()
    difficulty = request.form.get('difficulty', "medium")
    num_questions = request.form.get('num_questions', 5)

    
    if not file or not topic:
        return jsonify({"error": "❌ Please provide a valid PDF and topic."}), 400

    try:
        # Step 1: Extract text from PDF
        pdf_text = extract_pdf_text(file)

        # Step 2: Extract topic-specific content
        content = extract_topic_content(pdf_text, topic)
        if not content:
            return jsonify({"error": "⚠️ No relevant content found for the topic!"}), 400

        # Step 3: Extract keywords from content
        keywords = extract_keywords(content)

        # Step 4: Generate quiz
        quiz_output = generate_mcqs_bart(topic, keywords, content, int(num_questions), difficulty)

        # Step 5: Parse quiz output
        quiz_questions = []
        for q in quiz_output.strip().split("\n\n"):
            parts = q.strip().split("\n")
            if len(parts) >= 5:
                question = parts[0].replace("Q: ", "").strip()
                options = [parts[i][2:].strip() for i in range(1, 5)]
                quiz_questions.append({
                    "question": question,
                    "options": options
                })

        return jsonify({"quiz": quiz_questions})

    except Exception as e:
        print(f"Error generating quiz: {str(e)}")
        return jsonify({"error": "⚠️ Error fetching quiz. Try again!"}), 500




def generate_study_plan_with_bart(keywords, days_left):
    """Generate study plan using BART model."""
    prompt = f"""
You are an expert study planner.
Based on the following important topics extracted from my notes:

{', '.join(keywords)}

I have {days_left} days left before my exam.
Generate a very detailed day-by-day study plan.

- Prioritize important topics first.
- Mention exactly what to study each day.
- Include study tips and revision strategies.
- Be very structured and motivational.

Format it nicely day by day.
"""
    inputs = bart_tokenizer_quiz_exam(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = bart_model_quiz_exam.generate(inputs['input_ids'], max_length=250, num_return_sequences=1)
    study_plan = bart_tokenizer_quiz_exam.decode(outputs[0], skip_special_tokens=True)
    return study_plan

@app.route('/generate-strategy', methods=['POST'])
def generate_strategy():
    """Receive PDF, extract content, generate study plan."""
    if 'file' not in request.files:
        return jsonify({"error": "❌ No file part"}), 400
    file = request.files['file']
    days_left = int(request.form.get('days_left', 0))

    if not file or not days_left:
        return jsonify({"error": "❌ Please provide a valid PDF and number of days."}), 400

    try:
        
        pdf_text = extract_pdf_text(file)

        keywords = extract_keywords(pdf_text)

        
        study_plan = generate_study_plan_with_bart(keywords, days_left)

        if study_plan:
            return jsonify({"strategy": study_plan})
        else:
            return jsonify({"error": "⚠️ Failed to generate study plan."}), 500

    except Exception as e:
        print(f"Error generating strategy: {str(e)}")
        return jsonify({"error": "⚠️ Error generating strategy. Try again!"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
