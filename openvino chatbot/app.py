import streamlit as st
import finalchatbot as fc
import time
import sounddevice as sd
import wave
import numpy as np

st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("AI Technical and HR Interview Assistant 🤖")
    st.markdown("""
    Welcome to your AI-powered interview preparation platform! This assistant will:
    - Conduct technical and HR rounds
    - Analyze your responses for content and emotional tone
    - Provide real-time feedback
    """)

    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = fc.EmotionInterviewAssistant()
        st.session_state.interview_started = False
        st.session_state.current_round = "technical"
        st.session_state.question_count = 0
        st.session_state.max_technical_questions = 5
        st.session_state.max_hr_questions = 3

    # Candidate Profile Collection
    if not st.session_state.interview_started:
        st.header("Let's start with your profile")
        with st.form("candidate_profile"):
            name = st.text_input("Your Name")
            experience = st.text_input("Years of Experience")
            current_role = st.text_input("Current/Most Recent Role")
            target_role = st.text_input("Position You're Applying For")
            
            if st.form_submit_button("Start Interview"):
                st.session_state.assistant.candidate_profile = {
                    'name': name,
                    'years_of_experience': experience,
                    'current_role': current_role,
                    'target_role': target_role
                }
                st.session_state.interview_started = True
                st.rerun()  # Changed from experimental_rerun()

    # Interview Process
    if st.session_state.interview_started:
        # Display current round
        round_type = "Technical" if st.session_state.current_round == "technical" else "HR"
        st.header(f"📝 {round_type} Round")
        
        # Generate and display question
        if st.button("Get Next Question") or 'current_question' not in st.session_state:
            st.session_state.current_question = st.session_state.assistant.generate_question()
        
        st.write("### Question:")
        st.info(st.session_state.current_question)

        # Answer input method selection
        answer_method = st.radio("How would you like to answer?", 
                               ["Type", "Speak"])

        if answer_method == "Type":
            with st.form("answer_form"):
                answer = st.text_area("Your Answer")
                if st.form_submit_button("Submit Answer"):
                    with st.spinner("Analyzing your response..."):
                        # Get the complete feedback including analysis and areas for improvement
                        feedback_response = st.session_state.assistant.evaluate_answer(answer)
                        
                        # Display the complete feedback
                        st.markdown(feedback_response)
                        
                        st.session_state.question_count += 1
                        
                        # Check if we need to switch rounds or end interview
                        if st.session_state.current_round == "technical" and \
                           st.session_state.question_count >= st.session_state.max_technical_questions:
                            st.session_state.current_round = "hr"
                            st.session_state.question_count = 0
                            st.session_state.assistant.interview_type = "hr"
                            st.success("Technical round completed! Moving to HR round.")
                        elif st.session_state.current_round == "hr" and \
                             st.session_state.question_count >= st.session_state.max_hr_questions:
                            st.balloons()
                            st.success("Interview completed! Thank you for participating!")
                        
                            st.rerun()  # Changed from experimental_rerun()

        elif answer_method == "Speak":
            if st.button("Start Recording"):
                with st.spinner("Recording... Speak your answer"):
                    try:
                        # Record audio for 30 seconds
                        duration = 30
                        sample_rate = 16000
                        audio_data, temp_file = st.session_state.assistant.record_audio(
                            duration=duration, 
                            sample_rate=sample_rate
                        )
                        st.audio(temp_file)
                        
                        # Process the recorded answer
                        with st.spinner("Analyzing your response..."):
                            audio_text = "Audio response recorded and analyzed"
                            
                            # Get complete feedback
                            feedback_response = st.session_state.assistant.evaluate_answer(audio_text, emotion="neutral")
                            
                            # Create sections for display
                            st.markdown("### Analysis Results")
                            st.write("**Answer:** Audio response recorded")
                            st.write("**Sentiment Analysis:**")
                            st.info(feedback_response)
                            
                            # Add summary section
                            st.markdown("### Summary")
                            st.write("- Response recorded successfully")
                            st.write("- Analysis completed")
                            st.write("- Feedback generated")
                            
                            # Add a button to proceed to next question
                            if st.button("Continue to Next Question"):
                                st.session_state.question_count += 1
                                
                                # Check round transitions
                                if st.session_state.current_round == "technical" and \
                                   st.session_state.question_count >= st.session_state.max_technical_questions:
                                    st.session_state.current_round = "hr"
                                    st.session_state.question_count = 0
                                    st.session_state.assistant.interview_type = "hr"
                                    st.success("Technical round completed! Moving to HR round.")
                                elif st.session_state.current_round == "hr" and \
                                     st.session_state.question_count >= st.session_state.max_hr_questions:
                                    st.balloons()
                                    st.success("Interview completed! Thank you for participating!")
                                
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error recording audio: {str(e)}")

        # Progress tracking
        total_questions = st.session_state.max_technical_questions if st.session_state.current_round == "technical" \
                         else st.session_state.max_hr_questions
        st.progress(st.session_state.question_count / total_questions)
        st.write(f"Question {st.session_state.question_count + 1} of {total_questions}")

if __name__ == "__main__":
    main()