# Some functions from https://github.com/justinbodnar/political-compass

import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

def do_strike(agree, answer, axis, weight, x_coord, y_coord):
	"""
	Calculate the updated coordinates based on the user's answer.

	Args:
		agree (str): Direction of agreement ('+' or '-').
		answer (int): User's answer (1: Strongly Agree to 4: Strongly Disagree).
		axis (str): Axis affected by the question ('x' or 'y').
		weight (float): The weight of the question.
		x_coord (float): Current x-coordinate.
		y_coord (float): Current y-coordinate.

	Returns:
		tuple: Updated (x_coord, y_coord).
	"""
	strike = weight
	if 1 < answer < 4:
		strike /= 2.0

	# Adjust based on agreement or disagreement
	if answer < 3:
		if agree == "-":
			strike *= -1
	elif agree == "+":
		strike *= -1

	strike /= 2.0

	# Update the appropriate axis
	if axis == "y":
		y_coord += strike
	elif axis == "x":
		x_coord += strike

	return x_coord, y_coord


def compute_coord(df_results):
    x_coord = 0.0
    y_coord = 0.0

    for index, row in df_results.iterrows():
        answer_score = row['mean_scores']
        axis = row['axis']
        weight = row['units']
        agree = row['agree']

        if answer_score is not None:  # Only process if a valid score was obtained
            x_coord, y_coord = do_strike(agree, answer_score, axis, weight, x_coord, y_coord)

        else:
            print(f"Skipping row {index} due to missing score.")

    return x_coord, y_coord

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

def gen_compass_fr_en(filename_prefix="political_compass"):
    """
    Generate political compass charts for French and English responses and save as PNG files.
    Shows the political positions of different AI models based on their responses.
    """
    # Create figures for French and English
    fig_fr, ax_fr = plt.subplots(figsize=(12, 8))
    fig_en, ax_en = plt.subplots(figsize=(12, 8))
    
    # Load points from results.json
    with open('results.json', 'r') as f:
        results = json.load(f)
    
    # Define colors and markers for different model families
    model_styles = {
        'openai': {'color': '#4285F4', 'marker': 'o', 'name': 'OpenAI'},
        'deepseek': {'color': '#34A853', 'marker': 's', 'name': 'DeepSeek'},
        'x-ai': {'color': '#EA4335', 'marker': '^', 'name': 'X.AI'},
        'mistralai': {'color': '#FBBC05', 'marker': '*', 'name': 'Mistral AI'}
    }
    
    # Function to setup axis
    def setup_axis(ax, title):
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        
        # Add background colors for the quadrants
        ax.fill_betweenx([0, 10], 0, 10, color="#FFCCCC", alpha=0.2, zorder=0)  # Right-Authoritarian
        ax.fill_betweenx([0, 10], -10, 0, color="#CCCCFF", alpha=0.2, zorder=0)  # Left-Authoritarian
        ax.fill_betweenx([-10, 0], 0, 10, color="#CCFFCC", alpha=0.2, zorder=0)  # Right-Libertarian
        ax.fill_betweenx([-10, 0], -10, 0, color="#FFFFCC", alpha=0.2, zorder=0)  # Left-Libertarian
        
        # Draw gridlines
        ax.axhline(0, color='black', linewidth=1.0, zorder=1)  # Horizontal axis
        ax.axvline(0, color='black', linewidth=1.0, zorder=1)  # Vertical axis
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3, zorder=1)
        
        # Add axis labels and title
        ax.set_xlabel("Economic Left ← → Right", fontsize=20)
        ax.set_ylabel("Social Libertarian ← → Authoritarian", fontsize=20)
        ax.set_title(f"Political Compass - {title}", fontsize=25, pad=20)
        
        # Add quadrant labels
        label_offset = 0.98  # Slightly inside from the edges
        quadrants = [
            ('Left\nAuthoritarian', (-9, 9)),
            ('Right\nAuthoritarian', (9, 9)),
            ('Left\nLibertarian', (-9, -9)),
            ('Right\nLibertarian', (9, -9))
        ]
        
        for quad, pos in quadrants:
            ax.text(pos[0], pos[1], quad,
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=10,
                   alpha=0.5)
    
    # Setup both axes
    setup_axis(ax_fr, "French")
    setup_axis(ax_en, "English")
    
    # Plot points and create legend handles for French
    legend_elements_fr = []
    legend_elements_en = []
    
    for model_family in model_styles:
        style = model_styles[model_family]
        family_points_fr = [(name, coords) for name, coords in results.get('fr', {}).items() if name.startswith(model_family)]
        family_points_en = [(name, coords) for name, coords in results.get('en', {}).items() if name.startswith(model_family)]
        
        if family_points_fr:
            # Plot points for French
            for model_name, coords in family_points_fr:
                scatter = ax_fr.scatter(coords[0], coords[1], 
                                       c=style['color'], 
                                       marker=style['marker'], 
                                       s=200, 
                                       alpha=0.8,
                                       zorder=2)
                
            
            # Create legend element for French
            legend_elements_fr.append(plt.scatter([], [], 
                                                c=style['color'],
                                                marker=style['marker'],
                                                s=100,
                                                label=style['name']))
        
        if family_points_en:
            # Plot points for English
            for model_name, coords in family_points_en:
                scatter = ax_en.scatter(coords[0], coords[1], 
                                       c=style['color'], 
                                       marker=style['marker'], 
                                       s=200, 
                                       alpha=0.8,
                                       zorder=2)
                
            
            # Create legend element for English
            legend_elements_en.append(plt.scatter([], [], 
                                                c=style['color'],
                                                marker=style['marker'],
                                                s=100,
                                                label=style['name']))
    
    # Add legends
    ax_fr.legend(handles=legend_elements_fr,
                 title='AI models',
                 loc='center left',
                 fontsize=15,
                 title_fontsize=20,
                 bbox_to_anchor=(1, 0.5))
    
    ax_en.legend(handles=legend_elements_en,
                 title='AI models',
                 loc='center left',
                 fontsize=15,
                 title_fontsize=20,
                 bbox_to_anchor=(1, 0.5))
    
    # Adjust layouts to prevent label clipping
    fig_fr.tight_layout()
    fig_en.tight_layout()
    
    # Save the figures with high quality
    fr_filename = f"plots/{filename_prefix}_fr.png"
    en_filename = f"plots/{filename_prefix}_en.png"
    
    fig_fr.savefig(fr_filename, dpi=300, bbox_inches='tight')
    fig_en.savefig(en_filename, dpi=300, bbox_inches='tight')
    
    plt.close(fig_fr)
    plt.close(fig_en)
    
    return fr_filename, en_filename



def animate_compass(filename_prefix="political_compass_animation", save_as_gif=True):
    """
    Create an animated political compass plot showing transitions from English to French positions.
    Saves the animation as a .gif or .mp4 file.
    """

    # Load results
    with open("results.json", "r") as f:
        results = json.load(f)

    # Model styles
    model_styles = {
        'openai': {'color': '#4285F4', 'marker': 'o', 'name': 'OpenAI'},
        'deepseek': {'color': '#34A853', 'marker': 's', 'name': 'DeepSeek'},
        'x-ai': {'color': '#EA4335', 'marker': '^', 'name': 'X.AI'},
        'mistralai': {'color': '#FBBC05', 'marker': '*', 'name': 'Mistral AI'}
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(top=0.8)

    # Prepare data points
    points = []
    for model_family, style in model_styles.items():
        for model_name, en_coords in results.get("en", {}).items():
            if model_name.startswith(model_family) and model_name in results.get("fr", {}):
                fr_coords = results["fr"][model_name]
                points.append({
                    "model": model_name,
                    "family": model_family,
                    "style": style,
                    "start": np.array(en_coords),
                    "end": np.array(fr_coords)
                })

    num_frames = 30

    # Setup axis components that should persist
    def setup_background():
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_xlabel("Economic Left ← → Right", fontsize=16)
        ax.set_ylabel("Social Libertarian ← → Authoritarian", fontsize=16)

        # Add quadrant background using patches
        quadrant_colors = [
            {"xy": (0, 0), "color": "#FFCCCC"},    # Right-Authoritarian
            {"xy": (-10, 0), "color": "#CCCCFF"},  # Left-Authoritarian
            {"xy": (0, -10), "color": "#CCFFCC"},  # Right-Libertarian
            {"xy": (-10, -10), "color": "#FFFFCC"} # Left-Libertarian
        ]
        for quad in quadrant_colors:
            rect = Rectangle(quad["xy"], 10, 10, color=quad["color"], alpha=0.2, zorder=0)
            ax.add_patch(rect)

        # Axes
        ax.axhline(0, color='black', linewidth=1.0)
        ax.axvline(0, color='black', linewidth=1.0)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        # Quadrant labels
        quadrants = [
            ('Left\nAuthoritarian', (-9, 9)),
            ('Right\nAuthoritarian', (9, 9)),
            ('Left\nLibertarian', (-9, -9)),
            ('Right\nLibertarian', (9, -9))
        ]
        for label, (x, y) in quadrants:
            ax.text(x, y, label, ha='center', va='center', fontsize=10, alpha=0.5)

        # Legend
        legend_handles = []
        for style in model_styles.values():
            handle = plt.Line2D([0], [0], marker=style['marker'], color='w',
                                label=style['name'], markerfacecolor=style['color'],
                                markeredgecolor='k', markersize=10)
            legend_handles.append(handle)
        ax.legend(handles=legend_handles, title='AI Models', fontsize=12, title_fontsize=14, loc='upper right')

    def update(frame):
        ax.clear()
        setup_background()
        t = frame / (num_frames - 1)
        ax.set_title(f"\nFrom English (step 1) to French (step 30) \n\n Political Compass Transition (Step {frame+1}/{num_frames})", fontsize=20)

        for pt in points:
            interp = (1 - t) * pt["start"] + t * pt["end"]
            ax.scatter(interp[0], interp[1],
                       c=pt["style"]["color"],
                       marker=pt["style"]["marker"],
                       s=150,
                       edgecolors='k',
                       alpha=1.0)

    anim = FuncAnimation(fig, update, frames=num_frames, interval=150, repeat=False)

    output_file = f"plots/{filename_prefix}.gif" if save_as_gif else f"plots/{filename_prefix}.mp4"
    if save_as_gif:
        anim.save(output_file, writer='pillow', fps=5)
    else:
        anim.save(output_file, writer='ffmpeg', fps=5)

    plt.close(fig)
    return output_file


results = {}

questions = pd.read_csv("questions/questions_en_fr.csv")

for model in ["openai_gpt-4o", "deepseek_deepseek-chat-v3-0324", "x-ai_grok-beta", "mistralai_mistral-large-2411"]:
  for lang in ["en", "fr"]:
    df_scores = parse_responses(model, lang)
    df_scores['axis'] = questions['axis']
    df_scores['units'] = questions['units']
    df_scores['agree'] = questions['agree']
    if lang not in results:
      results[lang] = {}
    results[lang][model] = compute_coord(df_scores)

with open("results.json", "w") as fp:
    json.dump(results, fp)
   
results

gen_compass_fr_en()

animate_compass()

