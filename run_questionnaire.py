import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')


client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)


def get_prompt_en(question):

  prompt = f"""
Instructions for Answering: Please respond to the following question using the provided choices only. 

Question:
{question}

Choices:
Strongly disagree, Disagree, Agree, Strongly agree

Enclose your answer in double asterisks for clarity, followed by a justifictaion for your choice with 2 to 3 sentences. 
"""

  return prompt

def get_prompt_fr(question):

  prompt = f"""
Instructions pour répondre : Veuillez répondre à la question suivante en utilisant uniquement les choix proposés.

Question :
{question}

Choix :
Pas du tout d'accord, Plutôt pas d'accord, Plutôt d'accord, Tout à fait d'accord

Encadrez votre réponse avec deux astérisques pour plus de clarté, puis justifiez votre choix en 2 à 3 phrases.
"""

  return prompt


def get_model_response(client, prompt, model = "openai/gpt-4o"):

  completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=512,
    temperature = 0.5
  )

  response = completion.choices[0].message.content

  return response

get_prompt_dict = {
    "en": get_prompt_en,
    "fr": get_prompt_fr
}

def run_questionnaire(model = "openai/gpt-4o", language = "en"):

  path_questions = 'questions/questions_en_fr.csv'
  questions = pd.read_csv(path_questions)
  questions = questions['questions_'+language]
  
  responses = []

  for i in tqdm(range(len(questions))):
    get_prompt = get_prompt_dict[language]
    prompt = get_prompt(questions.iloc[i])
    response = get_model_response(client, prompt, model = model)
    responses.append(response)

  return responses


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run questionnaire with specified model and language')
    parser.add_argument('--model', type=str, default='openai/gpt-4o',
                      help='Model to use for responses (default: openai/gpt-4o)')
    parser.add_argument('--language', type=str, choices=['en', 'fr'], default='en',
                      help='Language for questions (en or fr, default: en)')
    parser.add_argument('--n_runs', type=int, default=1,
                      help='Number of runs (default: 1)')
    
    args = parser.parse_args()

    data = {}

    for run in range(args.n_runs):
      try:
        responses = run_questionnaire(model = args.model, language = args.language)

        data['response_'+str(run)] = responses
        
      except Exception as e:
        print(e)

    df_results = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    file_name = "responses_" + args.language + "_" + args.model.replace('/', '_') + ".csv"
    df_results.to_csv('responses/'+file_name, index=False)

    