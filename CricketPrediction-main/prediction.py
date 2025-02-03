import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import numpy as np

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Load the dataset
file_path = 'cricket_player_stats.csv'
data = load_and_preprocess_data(file_path)

# Example inputs for the models
team_name = "Australia"
format_name = "T20"
strike_rate = 90
runs_scored = 750
economy_rate = 3.8
wickets_taken = 30
matches_played = 10
bowling_average = 30
catches = 15
stumpings = 5
toss_decision = 1  # Example value, replace with actual if needed
result = 'Win'  # Example value, replace with actual if needed


# Train models for batsman and bowler performance
def train_models(data):
    # Initialize Linear Regression models
    batsman_model = LinearRegression()
    bowler_model = LinearRegression()
    
    # Fit batsman model
    batsmen_df = data[data['Primary Role'] == 'Batsman']
    X_batsman = batsmen_df[['Strike Rate', 'Runs Scored']]
    y_batsman = batsmen_df['Batting Average']
    batsman_model.fit(X_batsman, y_batsman)
    
    # Fit bowler model
    bowlers_df = data[data['Primary Role'] == 'Bowler']
    X_bowler = bowlers_df[['Economy Rate', 'Wickets Taken']]
    y_bowler = bowlers_df['Bowling Average']
    bowler_model.fit(X_bowler, y_bowler)
    
    return batsman_model, bowler_model

# Predict batting and bowling performance
def predict_batsman_performance(model, strike_rate, runs_scored):
    input_data = [[strike_rate, runs_scored]]
    predicted_performance = model.predict(input_data)
    random_variation = np.random.uniform(-0.5, 0.5)  # Random value between -5% and +5%
    predicted_performance_with_variation = predicted_performance * (1 + random_variation)
    return predicted_performance_with_variation

def predict_bowler_performance(model, economy_rate, wickets_taken):
    input_data = [[economy_rate, wickets_taken]]
    predicted_performance = model.predict(input_data)
    random_variation = np.random.uniform(-0.5, 0.5)  # Random value between -5% and +5%
    predicted_performance_with_variation = predicted_performance * (1 + random_variation)
    return predicted_performance_with_variation


batsman_model, bowler_model = train_models(data)

# Predict batting and bowling performance
# predicted_batting_average = predict_batsman_performance(batsman_model, strike_rate, runs_scored)
# predicted_bowling_average = predict_bowler_performance(bowler_model, economy_rate, wickets_taken)

# print(f"Predicted Batting Average: {predicted_batting_average}")
# print(f"Predicted Bowling Average: {predicted_bowling_average}")






# Train classification model for primary role prediction
def train_classification_model(data):
    player_classification_features = ['Matches Played', 'Runs Scored', 'Bowling Average', 'Catches', 'Stumpings']
    X = data[player_classification_features]
    y = data['Primary Role']

    # Encode the target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    # Feature scaling
    scaler_class = StandardScaler()
    X_train_class = scaler_class.fit_transform(X_train_class)
    X_test_class = scaler_class.transform(X_test_class)

    # Gradient Boosting Classifier
    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_classifier.fit(X_train_class, y_train_class)

    # Model evaluation
    y_pred_gb = gb_classifier.predict(X_test_class)
    
    return gb_classifier, scaler_class, le_target


# Predict primary role
def predict_primary_role(model, scaler, le_target, matches_played, runs_scored, bowling_average, catches, stumpings):
    input_data = pd.DataFrame({
        'Matches Played': [matches_played],
        'Runs Scored': [runs_scored],
        'Bowling Average': [bowling_average],
        'Catches': [catches],
        'Stumpings': [stumpings]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict using the trained model
    prediction_encoded = model.predict(input_data_scaled)
    prediction = le_target.inverse_transform(prediction_encoded)
    
    return prediction[0]

classification_model, scaler_class, le_target = train_classification_model(data)

# Predict primary role
# predicted_role = predict_primary_role(classification_model, scaler_class, le_target, matches_played, runs_scored, bowling_average, catches, stumpings)
# print(f"Predicted Primary Role: {predicted_role}")





# Train neural network for match outcome prediction
def train_neural_network(data):
    X_outcome = data[['Toss Decision', 'Matches Played', 'Runs Scored', 'Wickets Taken']]
    y_outcome = data['Result']

    # Encode features and target
    X_outcome = pd.get_dummies(X_outcome)
    le_outcome = LabelEncoder()
    y_outcome_encoded = le_outcome.fit_transform(y_outcome)

    # Split data
    X_train_outcome, X_test_outcome, y_train_outcome, y_test_outcome = train_test_split(X_outcome, y_outcome_encoded, test_size=0.2, random_state=42)

    # Train and evaluate the Neural Network model
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=1000, random_state=42)
    mlp_classifier.fit(X_train_outcome, y_train_outcome)

    # Evaluate the model
    y_pred_outcome = mlp_classifier.predict(X_test_outcome)
    
    return mlp_classifier, le_outcome, X_outcome.columns

# Predict match outcome
def predict_match_outcome(model, le_outcome, columns, toss_decision, matches_played, runs_scored, wickets_taken):
    input_data = pd.DataFrame({
        'Toss Decision': [toss_decision],
        'Matches Played': [matches_played],
        'Runs Scored': [runs_scored],
        'Wickets Taken': [wickets_taken]
    })

    # Convert categorical variables to dummy/indicator variables
    input_data = pd.get_dummies(input_data)

    # Align input data with the training data columns
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Predict using the trained model
    prediction_encoded = model.predict(input_data)
    prediction = le_outcome.inverse_transform(prediction_encoded)

    return prediction[0]

neural_network_model, le_outcome, columns = train_neural_network(data)

# Predict match outcome
# predicted_match_outcome = predict_match_outcome(neural_network_model, le_outcome, columns, toss_decision, matches_played, runs_scored, wickets_taken)
# print(f"Predicted Match Outcome: {predicted_match_outcome}")




# Select playing XI
def select_playing_11(dataset):
    num_batsmen = 3
    num_bowlers = 3
    num_allrounders = 2

    def create_target_variable(df):
        top_players = []
        for team in df['Team Name'].unique():
            team_df = df[df['Team Name'] == team]
            
            batsmen = team_df[team_df['Primary Role'] == 'Batsman'].sort_values(by='Runs Scored', ascending=False).head(num_batsmen)
            bowlers = team_df[team_df['Primary Role'] == 'Bowler'].sort_values(by='Wickets Taken', ascending=False).head(num_bowlers)
            allrounders = team_df[team_df['Primary Role'] == 'All-Rounder'].sort_values(by='Runs Scored', ascending=False).head(num_allrounders)
            
            selected_players = pd.concat([batsmen, bowlers, allrounders])
            remaining_spots = 11 - len(selected_players)
            
            if remaining_spots > 0:
                remaining_players = team_df[~team_df.index.isin(selected_players.index)].head(remaining_spots)
                selected_players = pd.concat([selected_players, remaining_players])
            
            top_players.append(selected_players)
        
        return pd.concat(top_players)

    dataset['Selected_in_Playing11'] = dataset.index.isin(create_target_variable(dataset).index)

    # Encode categorical variables
    label_encoder_team = LabelEncoder()
    label_encoder_format = LabelEncoder()

    # Fit the encoders on all data
    dataset['Team Name Encoded'] = label_encoder_team.fit_transform(dataset['Team Name'])
    dataset['Format Encoded'] = label_encoder_format.fit_transform(dataset['Format'])

    # Features and target variable
    X = dataset[['Team Name Encoded', 'Format Encoded', 'Strike Rate', 'Runs Scored', 'Wickets Taken', 'Economy Rate']]
    y = dataset['Selected_in_Playing11']

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = xgb_model.predict(X_test)

    # Function to predict the best playing 11
    def predict_playing11(team_name, format_name, num_batsmen=3, num_bowlers=3, num_allrounders=2):
        # Encode the input team and format
        if team_name not in label_encoder_team.classes_:
            label_encoder_team.classes_ = np.append(label_encoder_team.classes_, team_name)
        if format_name not in label_encoder_format.classes_:
            label_encoder_format.classes_ = np.append(label_encoder_format.classes_, format_name)
            
        encoded_team = label_encoder_team.transform([team_name])[0]
        encoded_format = label_encoder_format.transform([format_name])[0]
        
        # Extract the relevant players from the dataset
        relevant_players = dataset[(dataset['Team Name Encoded'] == encoded_team) & (dataset['Format Encoded'] == encoded_format)]
        
        # Sort and select players based on the provided criteria
        sorted_batsmen = relevant_players[relevant_players['Primary Role'] == 'Batsman'].sort_values(by='Runs Scored', ascending=False).head(num_batsmen)
        sorted_bowlers = relevant_players[relevant_players['Primary Role'] == 'Bowler'].sort_values(by='Wickets Taken', ascending=False).head(num_bowlers)
        sorted_allrounders = relevant_players[relevant_players['Primary Role'] == 'All-Rounder'].sort_values(by='Runs Scored', ascending=False).head(num_allrounders)
        
        playing_11 = pd.concat([sorted_batsmen, sorted_bowlers, sorted_allrounders])
        
        # Fill remaining spots with top performers not already selected
        remaining_spots = 11 - len(playing_11)
        if remaining_spots > 0:
            remaining_players = relevant_players[~relevant_players.index.isin(playing_11.index)].sort_values(by='Runs Scored', ascending=False).head(remaining_spots)
            playing_11 = pd.concat([playing_11, remaining_players])

        return playing_11

    return xgb_model, scaler, label_encoder_team, label_encoder_format, predict_playing11

# Train the models
xgb_model, scaler, label_encoder_team, label_encoder_format, predict_playing11 = select_playing_11(data)

#predicted_playing_11 = predict_playing11(team_name, format_name)
#print("Predicted Playing XI:")
#print(predicted_playing_11[['Player Name', 'Primary Role']])

