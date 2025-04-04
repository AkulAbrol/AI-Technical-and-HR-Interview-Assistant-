import os
import numpy as np
import torch
import torchaudio
import openvino as ov
from openvino.runtime import Core, Tensor, Type
import speech_recognition as sr
from transformers import AutoProcessor, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForAudioClassification
import sounddevice as sd
import librosa
import time
from scipy.io.wavfile import write
import wave
import requests
import json

API_KEY = "AIzaSyC_0pIA4PD94ErXmkqAM0Z24mXpzHWTr14"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + API_KEY

class EmotionInterviewAssistant:
    def __init__(self):
        self.core = Core()
        self.recognizer = sr.Recognizer()
        self.interview_type = "technical"  # Start with technical round
        self.candidate_profile = {}
        self.previous_answers = []
        
        # Initialize emotion recognition model
        self.emotion_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-speech-commands-v2")
        self.emotion_model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-speech-commands-v2")
        self.emotion_labels = ['neutral', 'happy', 'angry', 'sad']

        # Initialize sentiment analysis model
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        
        # Convert models to OpenVINO format
        self._convert_models_to_openvino()

    def set_candidate_profile(self):
        """Get candidate's basic information"""
        print("\nBefore we begin, please provide some information:")
        self.candidate_profile['name'] = input("Your name: ")
        self.candidate_profile['years_of_experience'] = input("Years of experience: ")
        self.candidate_profile['current_role'] = input("Current/Most recent role: ")
        self.candidate_profile['target_role'] = input("Position you're applying for: ")
        print("\nThank you! Let's begin with the technical round.")

    def generate_question(self):
        """Generate next question using Gemini API"""
        headers = {
            "Content-Type": "application/json"
        }

        # Create context based on previous answers and interview type
        context = f"Candidate Profile: {json.dumps(self.candidate_profile)}\n"
        context += f"Previous Answers: {json.dumps(self.previous_answers)}\n"
        
        if self.interview_type == "technical":
            prompt = f"""You are a technical interviewer. Based on the candidate's profile for {self.candidate_profile['target_role']} position:
            1. Generate a relevant technical question
            2. The question should be specific to their role and experience level
            3. Focus on practical scenarios they might encounter
            Previous questions asked: {self.previous_answers}
            Generate only the question, no other text."""
        else:  # HR round
            prompt = f"""You are an HR interviewer. Based on the candidate's profile and previous technical round performance:
            1. Generate a behavioral or situational question
            2. Focus on soft skills, team collaboration, and culture fit
            3. Make it relevant to their experience level and role
            Previous questions asked: {self.previous_answers}
            Generate only the question, no other text."""

        data = {
            "contents": [{
                "parts": [{
                    "text": context + "\n" + prompt
                }]
            }]
        }

        try:
            response = requests.post(API_URL, headers=headers, json=data)
            response.raise_for_status()
            question = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            self.previous_answers.append({"question": question, "type": self.interview_type})
            return question
        except Exception as e:
            print(f"Error generating question: {e}")
            return None

    def evaluate_answer(self, answer, emotion="neutral"):
        """Evaluate the answer and provide feedback"""
        sentiment = self.analyze_sentiment(answer)
        
        # Generate feedback using Gemini API
        headers = {
            "Content-Type": "application/json"
        }

        prompt = f"""As an AI interviewer, evaluate this response:
        Question: {self.previous_answers[-1]['question']}
        Answer: {answer}
        Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})
        Emotion: {emotion}
        Interview Type: {self.interview_type}

        Provide a brief, constructive feedback focusing on:
        1. Content quality
        2. Emotional tone appropriateness
        3. Areas for improvement
        Keep the feedback professional and encouraging."""

        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }

        try:
            response = requests.post(API_URL, headers=headers, json=data)
            response.raise_for_status()
            feedback = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e:
            print(f"Error generating feedback: {e}")
            feedback = self._generate_basic_feedback(sentiment, emotion)

        # Create complete output string with proper formatting
        complete_output = f"""Analysis Results:
        Answer: {answer}
        Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})
        Detected Emotion: {emotion}

        **Feedback:**
        {feedback}"""

        print(complete_output)  # Print to terminal
        time.sleep(7)  # Wait for 7 seconds to read terminal output
        return complete_output  # Return the complete output for frontend display

    def _generate_basic_feedback(self, sentiment, emotion):
        """Generate basic feedback when API is not available"""
        if sentiment['label'] == "POSITIVE":
            if emotion in ['happy', 'neutral']:
                return "Your answer was positive and your emotional tone matched well!"
            else:
                return "While your answer was positive, your emotional tone might need adjustment."
        else:
            if emotion in ['sad', 'angry']:
                return "Consider rephrasing your answer more positively and maintaining a more confident tone."
            else:
                return "Consider rephrasing your answer more positively, though your emotional tone was appropriate."

    def _convert_models_to_openvino(self):
        try:
            # Convert emotion model to OpenVINO format
            dummy_input = torch.randn(1, 1, 128, 128)
            emotion_model_ov = ov.convert_model(self.emotion_model, example_input=dummy_input)
            
            # Convert sentiment model to OpenVINO format
            dummy_text = self.tokenizer("dummy text", return_tensors="pt", padding=True, truncation=True)
            sentiment_model_ov = ov.convert_model(
                self.sentiment_model,
                example_input=(dummy_text['input_ids'], dummy_text['attention_mask'])
            )
            
            # Compile the OpenVINO models
            self.emotion_compiled_model = self.core.compile_model(emotion_model_ov)
            self.sentiment_compiled_model = self.core.compile_model(sentiment_model_ov)
            
            # Get output layers
            self.emotion_output_layer = self.emotion_compiled_model.output(0)
            self.sentiment_output_layer = self.sentiment_compiled_model.output(0)
            
            print("Successfully converted and loaded models with OpenVINO")
            self.use_openvino = True
        except Exception as e:
            print(f"Error converting models to OpenVINO: {e}")
            print("Falling back to PyTorch models")
            self.use_openvino = False

    def analyze_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        if self.use_openvino:
            try:
                input_ids = Tensor(inputs['input_ids'].numpy())
                attention_mask = Tensor(inputs['attention_mask'].numpy())
                results = self.sentiment_compiled_model([input_ids, attention_mask])[self.sentiment_output_layer]
                probabilities = torch.nn.functional.softmax(torch.from_numpy(results), dim=-1)
            except Exception as e:
                print(f"OpenVINO inference failed, falling back to PyTorch: {e}")
                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        else:
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        label = "POSITIVE" if probabilities[0][1] > probabilities[0][0] else "NEGATIVE"
        score = probabilities[0][1].item() if label == "POSITIVE" else probabilities[0][0].item()
        return {"label": label, "score": score}

    def record_audio(self, duration=10, sample_rate=16000):
        """Record audio from microphone"""
        try:
            print(f"Recording for {duration} seconds...")
            # Record audio
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()  # Wait for recording to complete
            
            # Save the recorded audio to a temporary file
            temp_file = f"temp_recording_{int(time.time())}.wav"
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            return audio_data.flatten(), temp_file
        except Exception as e:
            print(f"Error recording audio: {e}")
            raise

    def preprocess_audio(self, audio_data, sample_rate=16000):
        """Preprocess audio data for model inference"""
        try:
            # Convert to float32 for processing
            audio_float = audio_data.astype(np.float32) / 32768.0  # Normalize 16-bit audio to float

            if len(audio_float.shape) > 1:
                audio_float = audio_float.mean(axis=1)
            
            # Resample if necessary
            if sample_rate != 16000:
                audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)
                
            return audio_float
        except Exception as e:
            print(f"Error in audio preprocessing: {e}")
            return audio_data.astype(np.float32) / 32768.0  # Return normalized data as fallback

    def analyze_emotion(self, audio_data):
        try:
            inputs = self.emotion_processor(audio_data, sampling_rate=16000, return_tensors="pt")
            
            if self.use_openvino:
                try:
                    input_values = Tensor(inputs.input_values.numpy())
                    results = self.emotion_compiled_model([input_values])[self.emotion_output_layer]
                    predictions = torch.nn.functional.softmax(torch.from_numpy(results), dim=-1)
                except Exception as e:
                    print(f"OpenVINO inference failed, falling back to PyTorch: {e}")
                    with torch.no_grad():
                        outputs = self.emotion_model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            else:
                with torch.no_grad():
                    outputs = self.emotion_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            pred_idx = predictions.argmax().item()
            emotions = {0: 'neutral', 1: 'happy', 2: 'angry', 3: 'sad'}
            predicted_emotion = emotions.get(pred_idx % 4, 'neutral')
            
            return predicted_emotion
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
            return 'neutral'

if __name__ == "__main__":
    assistant = EmotionInterviewAssistant()
    
    print("Welcome to the AI Technical and HR Interview Assistant!")
    print("\nThis assistant will conduct a technical round followed by an HR round.")
    print("Your responses will be analyzed for both content and emotional tone.")
    
    # Get candidate profile
    assistant.set_candidate_profile()
    
    # Technical Round
    print("\n=== Technical Round ===")
    for i in range(5):  # 5 technical questions
        question = assistant.generate_question()
        if not question:
            print("Error generating question. Using default question.")
            question = "Tell me about your technical background."
            
        print(f"\nQuestion {i+1}: {question}")
        print("\nWould you like to:")
        print("1. Type your answer")
        print("2. Speak your answer")
        print("3. Skip this question")
        
        choice = input("\nEnter your choice (1, 2, or 3): ")
        
        if choice == "3":
            continue
        elif choice == "1":
            text_answer = input("\nEnter your answer: ")
            assistant.evaluate_answer(text_answer, emotion="neutral")
        elif choice == "2":
            try:
                print("\nGet ready to speak...")
                audio_data, temp_file = assistant.record_audio()
                processed_audio = assistant.preprocess_audio(audio_data)
                emotion = assistant.analyze_emotion(processed_audio)
                
                with sr.AudioFile(temp_file) as source:
                    print("Processing audio...")
                    audio = assistant.recognizer.record(source)
                    try:
                        text = assistant.recognizer.recognize_google(audio)
                        print(f"\nTranscribed text: {text}")
                        assistant.evaluate_answer(text, emotion)
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                    except sr.RequestError as e:
                        print(f"Could not request results; {e}")
                    
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    # HR Round
    print("\n=== HR Round ===")
    assistant.interview_type = "hr"
    
    for i in range(3):  # 3 HR questions
        question = assistant.generate_question()
        if not question:
            print("Error generating question. Using default question.")
            question = "Tell me about yourself."
            
        print(f"\nQuestion {i+1}: {question}")
        print("\nWould you like to:")
        print("1. Type your answer")
        print("2. Speak your answer")
        print("3. Skip this question")
        
        choice = input("\nEnter your choice (1, 2, or 3): ")
        
        if choice == "3":
            continue
        elif choice == "1":
            text_answer = input("\nEnter your answer: ")
            assistant.evaluate_answer(text_answer, emotion="neutral")
        elif choice == "2":
            try:
                print("\nGet ready to speak...")
                audio_data, temp_file = assistant.record_audio()
                processed_audio = assistant.preprocess_audio(audio_data)
                emotion = assistant.analyze_emotion(processed_audio)
                
                with sr.AudioFile(temp_file) as source:
                    print("Processing audio...")
                    audio = assistant.recognizer.record(source)
                    try:
                        text = assistant.recognizer.recognize_google(audio)
                        print(f"\nTranscribed text: {text}")
                        assistant.evaluate_answer(text, emotion)
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                    except sr.RequestError as e:
                        print(f"Could not request results; {e}")
                    
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    print("\nInterview complete! Thank you for participating!")
    print("Good luck with your application!")
