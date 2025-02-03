# app.py

from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from prediction import predict_batsman_performance, predict_bowler_performance, batsman_model, bowler_model, predict_primary_role, classification_model, scaler_class, le_target, predict_match_outcome, neural_network_model, le_outcome, columns, predict_playing11
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__, static_folder='static')

# Load your dataset here
data = pd.read_csv('cricket_player_stats.csv')

def plot_stats(player_data, color):
    # Filter numeric columns for plotting
    numeric_data = player_data.select_dtypes(include=['number'])
    if numeric_data.empty:
        return None
    
    # Create bar chart for average stats   
    fig, ax = plt.subplots()
    numeric_data.mean().plot(kind='bar', color=color, ax=ax)
    ax.set_title('Average Stats')
    ax.set_ylabel('Average Value')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return img_base64

def pca_plot(player1, player2):
    # Prepare data for PCA
    features = ['Matches Played', 'Strike Rate', 'Runs Scored', 'Wickets Taken', 'Catches', 'Stumpings', 'Batting Average', 'Bowling Average', 'Economy Rate']
    
    # Ensure we only include relevant features and drop rows with missing values
    pca_data = data[features].dropna()
    unique_players = data['Player Name'].unique()
    
    if player1 not in unique_players or player2 not in unique_players:
        return None
    
    # Standardize features
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_data_scaled)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Player Name'] = data['Player Name']
    
    # Filter PCA data for selected players
    pca_df_player1 = pca_df[pca_df['Player Name'] == player1]
    pca_df_player2 = pca_df[pca_df['Player Name'] == player2]

    fig, ax = plt.subplots()
    ax.scatter(pca_df['PC1'], pca_df['PC2'], label='Players', alpha=0.5, color='grey')
    if not pca_df_player1.empty:
        ax.scatter(pca_df_player1['PC1'], pca_df_player1['PC2'], label=f'{player1}', color='#1f77b4')
    if not pca_df_player2.empty:
        ax.scatter(pca_df_player2['PC1'], pca_df_player2['PC2'], label=f'{player2}', color='#ff7f0e')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    pca_img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return pca_img_base64

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == "POST":
        player1 = request.form['player1']
        player2 = request.form['player2']

        if player1 == player2:
            return render_template('compare.html', error="Please select different players.")

        player1_data = data[data['Player Name'] == player1]
        player2_data = data[data['Player Name'] == player2]

        player1_img = plot_stats(player1_data, '#1f77b4')
        player2_img = plot_stats(player2_data, '#ff7f0e')
        pca_img = pca_plot(player1, player2)

        # Prepare detailed statistics for each player
        player1_stats = player1_data.iloc[0].to_dict()
        player2_stats = player2_data.iloc[0].to_dict()

        return render_template('compare.html', 
                            player1=player1,
                            player2=player2,
                            player1_img=player1_img,
                            player2_img=player2_img,
                            pca_img=pca_img,
                            player1_stats=player1_stats,
                            player2_stats=player2_stats)
    players = data['Player Name'].unique()
    print(players)
    return render_template('compare_input.html', players=list(players))


@app.route('/batting-bowling', methods=['GET', 'POST'])
def battingBowlingForm():
    if request.method == 'POST':
        prediction_type = request.form.get('prediction_type')

        if prediction_type == 'batting' and request.form.get('strike_rate') and request.form.get('runs_scored'):
            strike_rate = float(request.form.get('strike_rate'))
            runs_scored = float(request.form.get('runs_scored'))
            predicted_batting_average = predict_batsman_performance(batsman_model, strike_rate, runs_scored)
            return render_template('batting_bowling_result.html', prediction_type=prediction_type, result=predicted_batting_average)

        elif prediction_type == 'bowling' and request.form.get('economy_rate') and request.form.get('wickets_taken'):
            economy_rate = float(request.form.get('economy_rate'))
            wickets_taken = float(request.form.get('wickets_taken'))
            predicted_bowling_average = predict_bowler_performance(bowler_model, economy_rate, wickets_taken)
            return render_template('batting_bowling_result.html', prediction_type=prediction_type, result=predicted_bowling_average)
    
    return render_template('batting_bowling.html')

# Prediction route
@app.route('/primary-role', methods=['GET', 'POST'])
def primaryRole():
    prediction = None
    if request.method == 'POST':
        matches_played = int(request.form['matches_played'])
        runs_scored = int(request.form['runs_scored'])
        bowling_average = float(request.form['bowling_average'])
        catches = int(request.form['catches'])
        stumpings = int(request.form['stumpings'])

        prediction = predict_primary_role(classification_model, scaler_class, le_target, matches_played, runs_scored, bowling_average, catches, stumpings)

    return render_template('primary_role.html', prediction=prediction)


@app.route('/match-outcome', methods=['GET', 'POST'])
def matchOutcome():
    if request.method == 'POST':
        toss_decision = request.form['toss_decision']
        matches_played = int(request.form['matches_played'])
        runs_scored = int(request.form['runs_scored'])
        wickets_taken = int(request.form['wickets_taken'])

        predicted_match_outcome = predict_match_outcome(
            neural_network_model, le_outcome, columns, toss_decision, matches_played, runs_scored, wickets_taken
        )
        return render_template('match_outcome.html', prediction=predicted_match_outcome)

    return render_template('match_outcome.html', prediction=None)

@app.route('/playing-xi', methods=['GET', 'POST'])
def playingXi():
    prediction = None
    if request.method == 'POST':
        team_name = request.form['team_name']
        format_name = request.form['format_name']
        predicted_playing_11 = predict_playing11(team_name, format_name)
        prediction = predicted_playing_11[['Player Name', 'Primary Role']].to_dict(orient='records')
    return render_template('playing_xi.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
