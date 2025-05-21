import gradio as gr
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

# Load questions
questions = pd.read_csv('questions/questions_en_fr.csv')

def get_score(response):

    response_split = response.split("**")
    if len(response_split) > 1:
        score = response_split[1]
    else:
        response_split = response.split("*")
        if len(response_split) > 1:
            score = response_split[1]
        else:
            return None

    if pd.isna(score):
        return None
    agreement_map = {
        'strongly disagree': 4,
        'pas du tout d’accord': 4,
        "pas du tout d'accord": 4,
        'disagree': 3,
        'plutôt pas d’accord': 3,
        "plutôt pas d'accord": 3,
        'plutôt pas d’acoord': 3,
        "plutôt pas d'acoord": 3,
        'agree': 2,
        'plutôt d’accord': 2,
        "plutôt d'accord": 2,
        'strongly agree': 1,
        'tout à fait d’accord': 1,
        "tout à fait d'accord": 1,

    }
    score = score.lower()
    for phrase, num in agreement_map.items():
        if phrase == score:
            return num
    return None

def parse_responses(model, language):
    
    path = f"responses/responses_{language}_{model.replace('/', '_')}.csv"
    df = pd.read_csv(path)

    n = df.shape[1]

    for i in range(n):
        scores = df['response_'+str(i)].apply(get_score)
        df['score_'+str(i)] = scores

    df_scores = df[['score_'+str(i) for i in range(n)]]

    df['mean_scores'] = df_scores.mean(axis=1, skipna=True)
    df['std_scores'] = df_scores.std(axis=1, skipna=True)
    
    return df


def create_column_content(question_id, lang='fr'):
    # Get questions dataset based on language
    question = questions['questions_fr' if lang == 'fr' else 'questions_en'].iloc[question_id]
    
    # Load responses for each model
    model_responses = {}
    model_scores = {}
    
    # Define model families and their specific models
    models = ["openai_gpt-4o", "deepseek_deepseek-chat-v3-0324", "x-ai_grok-beta", "mistralai_mistral-large-2411"]
    
    # Load responses from CSV files
    for model in models:
            try:
                df = parse_responses(model, lang)
                if not df.empty:
                    response = df[[col for col in df if col.startswith('response_')]]
                    score = df[[col for col in df if col.startswith('score_')]]
                    model_responses[model] = response.iloc[question_id].values
                    model_scores[model] = score.iloc[question_id].values
            except (FileNotFoundError, pd.errors.EmptyDataError):
                continue
    
    # Create visualization
    fig = go.Figure()
    
    # Define colors for model families
    colors = {
        'openai': '#4285F4',
        'deepseek': '#34A853',
        'x-ai': '#EA4335',
        'mistralai': '#FBBC05'
    }
    
    # Process data for visualization
    for model_name, scores in model_scores.items():
        family = model_name.split('_')[0]
        model_display_name = model_name.split('_')[1]
        
        # Filter out None values
        valid_scores = [s for s in scores if pd.notna(s)]
        
        if valid_scores:
            # Add box plot for each model
            fig.add_trace(go.Box(
                x=valid_scores,
                name=model_display_name,
                marker_color=colors[family],
                boxpoints='all',  # Show all points
                jitter=0.3,  # Add jitter to points
                pointpos=-1.8,  # Position points to the left of the box
                boxmean=True  # Show mean line
            ))
    
    # Update layout for better visualization
    fig.update_layout(
        title=dict(
            text="Score distribution (per model, from 1 to 10)",
            font=dict(size=16),
            xref='paper',
            x=0
        ),
        xaxis=dict(
            title='Score (1: Strongly agree to 4: Strongly disagree)',
            range=[0, 5],  # Slightly wider than data range for better visibility
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            title='Models',
            gridcolor='lightgray'
        ),
        showlegend=False,
        height=400,
        width=800,
        plot_bgcolor='white'
    )
    
    # Create markdown content with responses
    md_content = f"### Model Responses for Question:\n**{question}**\n\n"
    
    for model_name, responses in model_responses.items():
        md_content += f"#### {model_name} \n"
        for i in range (len(responses)):
            md_content += f"#### Run {i}: \n\n {responses[i]}\n\n"
    
    return fig, md_content

def update_content(question_en, question_fr):
    # Get indices from questions
    idx_en = questions[questions['questions_en'] == question_en].index[0]
    idx_fr = questions[questions['questions_fr'] == question_fr].index[0]
    
    fig_en, answers_en_md = create_column_content(idx_en, 'en')
    fig_fr, answers_fr_md = create_column_content(idx_fr, 'fr')
    
    return fig_en, fig_fr, answers_en_md, answers_fr_md

css = """
h1, h3 {
    text-align: center;
    display:block;
}
"""

# Create the interface
with gr.Blocks(theme=gr.themes.Soft(), css=css) as interface:
    gr.Markdown('# How do AI political biases differ')
    gr.Markdown('# between English and French?')
    
    gr.Markdown("---")
    gr.Markdown("## Overview")
    
    with gr.Row():          

        with gr.Column():
            gr.Markdown('#### What is this ?')
            gr.Markdown("""
The animation shows how generative AI models shift on the political compass when switching from English to French prompts.

The question bank is a set of 62 political questions from the [Political Compass](https://politicalcompass.org/).

Four models were tested:
- OpenAI GPT-4o
- DeepSeek DeepSeek-chat-v3-0324
- X-ai Grok-beta
- MistralAI Mistral-large-2411

All questions were asked three times, and the scores are averaged to compute the final political compass.

Details of the model answers and scores are available below.
            """)

        with gr.Column():
            #gr.Markdown('#### French political compass')
            img_overview = gr.Image("plots/political_compass_animation.gif", type="filepath", label="Animated Political Compass")
            
    # Political compass visualizations
    gr.Markdown("---")
    gr.Markdown("## Political Compass")
    gr.Markdown("""
These two maps show the political compass of the models.
            
            """)
    with gr.Row():          

        with gr.Column():
            #gr.Markdown('#### French political compass')
            img_en = Image.open("plots/political_compass_en.png")
            compass_plot_en = gr.Image(value=img_en, type="pil", show_label=False)
        
        with gr.Column():
            #gr.Markdown('#### English political compass')
            img_fr = Image.open("plots/political_compass_fr.png")
            compass_plot_fr = gr.Image(value=img_fr, type="pil", show_label=False)
    
    
    gr.Markdown("---")
    gr.Markdown("## Question bank and model responses")
    with gr.Row(): 
        # Get initial figures and content
        initial_en_fig, initial_fr_fig, initial_en_md, initial_fr_md = update_content(
            questions['questions_en'].iloc[22],
            questions['questions_fr'].iloc[22]
        )

        with gr.Row():
            # Left column for English questions
            with gr.Column():
                gr.Markdown('### English Questions')
                dropdown_en = gr.Dropdown(
                choices=questions['questions_en'].tolist(),
                label='Select a question in English',
                value=questions['questions_en'].iloc[22]
                )
                plot_en = gr.Plot(value=initial_en_fig, label='English Responses')
                answers_en = gr.Markdown(value=initial_en_md, label='Response Details')
    
            # Right column for French questions
            with gr.Column():
                gr.Markdown('### French Questions')
                dropdown_fr = gr.Dropdown(
                    choices=questions['questions_fr'].tolist(),
                label='Select a question in French',
                value=questions['questions_fr'].iloc[22]
                )
                plot_fr = gr.Plot(value=initial_fr_fig, label='French Responses')
                answers_fr = gr.Markdown(value=initial_fr_md, label='Response Details')
            
        # Update content when dropdowns change
        def sync_questions(selected_question, source_lang):
            # Find the index of the selected question
            if source_lang == 'fr':
                idx = questions[questions['questions_fr'] == selected_question].index[0]
                corresponding_question = questions['questions_en'].iloc[idx]
            else:
                idx = questions[questions['questions_en'] == selected_question].index[0]
                corresponding_question = questions['questions_fr'].iloc[idx]
            
            # Get content for both languages
            fig_fr, fig_en, md_fr, md_en = update_content(questions['questions_fr'].iloc[idx], questions['questions_en'].iloc[idx])
        
            return [
                corresponding_question,  # Update other dropdown
                fig_fr, fig_en,         # Update plots
                md_fr, md_en           # Update markdown content
            ]
    
        # Update content when French dropdown changes
        dropdown_fr.change(
            fn=lambda q: sync_questions(q, 'fr'),
            inputs=[dropdown_fr],
            outputs=[dropdown_en, plot_fr, plot_en, answers_fr, answers_en]
        )
    
        # Update content when English dropdown changes
        dropdown_en.change(
            fn=lambda q: sync_questions(q, 'en'),
            inputs=[dropdown_en],
            outputs=[dropdown_fr, plot_fr, plot_en, answers_fr, answers_en]
        )
    
    
    gr.Markdown("---")
    gr.Markdown("## About")
    with gr.Row(): 

        gr.Markdown("""
Made with ❤️ by [Yann-Aël Le Borgne](https://www.linkedin.com/in/yannaelb/)

Source code: [GitHub](https://github.com/yalb/ai-political-bias)

Inspired by:
- [David Rozado's work](https://davidrozado.substack.com/p/new-results-of-state-of-the-art-llms)
- [Political Compass](https://politicalcompass.org/)
- [TrackingAI](https://trackingai.io/)
- [SpeechMap](https://speechmap.ai/)
        """)


if __name__ == '__main__':
    interface.launch(share=True)