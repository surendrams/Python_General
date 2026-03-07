import json

from openai import OpenAI
from typing import Dict, Any

client = OpenAI(api_key="")


def generate_challenge(subject, difficulty, topics=None):
    system_prompt = """You are an expert challenge/question creator. 
    Your task is to generate a novel and original question with multiple-choice answers on a given subject, 
    with appropriate for the specified difficulty level.  
    The question should not be a rehash of existing or commonly known questions.

    For easy questions: Focus on fundamental concepts, basic definitions, or simple applications of the subject matter.
    For medium questions: Cover intermediate concepts, common applications, or relationships between different aspects of the subject.
    For hard questions: Include advanced topics, complex scenarios, nuanced understanding, or critical analysis.

    The question should be clear, concise, and test a specific piece of knowledge or skill.  All options should be plausible but only one should be clearly the most correct or best answer.

    Return the challenge in the following JSON structure:

    ```json
    {
        "title": "The question title",
        "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
        "correct_answer_id": 0, // Index of the correct answer (0-3)
        "explanation": "Detailed explanation of why the correct answer is right.  Explain why the other options are incorrect or less optimal."
    }
    ```
    """

    user_prompt = f"Generate a {difficulty} difficulty {subject} challenge question."

    if topics:
        user_prompt += f" Focus on these topics: {', '.join(topics)}."

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",  # Specify a model - important for reproducibility
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7  # Adjust for creativity vs. accuracy
        )

        content = response.choices[0].message.content
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Raw content: {content}") # Print the raw content for debugging
                return None
        else:
            print("No content received from the API.")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Example Usage
if __name__ == '__main__':
    #challenge = generate_challenge(subject="World History", difficulty="medium", topics=["The Cold War", "The Russian Revolution"])
    challenge = generate_challenge(subject="Python", difficulty="medium", topics=[])

    if challenge:
        print(json.dumps(challenge))
    else:
        print("Failed to generate challenge.")
