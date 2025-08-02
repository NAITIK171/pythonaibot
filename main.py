import requests, time, os, random
from datetime import datetime, timezone
import math 

# ==== Telegram Setup ====
# For local terminal, we'll prompt the user for input
# If you want to avoid typing these every time, you can set them as system environment variables
# or hardcode them here (NOT recommended for security, especially if sharing code).
# Get bot token and chat ID from environment variables
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")
use_bot = bool(bot_token and chat_id)

if not use_bot:
    print("WARNING: Telegram bot token or chat ID not set. Bot functionality will be disabled.")


# ==== Sticker IDs ====
STARTUP_STICKER = "CAACAgUAAxkBAAEBZ59oavRmwgnu5YDciQ6NK-YWOCZ8CgACVBMAAoI5aFWhFY64_sGyZzYE"
WIN_STICKER = "CAACAgUAAxkBAAEBZm9oaPeI3BDlen_9cyQ8hSpCQCAQuwACLBAAAm1SQFUM-uASNVLKVzYE"
LOSS_STICKER = "CAACAgUAAxkBAAEBZ5hoavNEoWBMwIxSW1hJntbIFIAjXAACGg0AAnZsKFdSdl8cfAABlrM2BA"

# ==== API CONFIG (from HTML) ====
API_URL = "https://kbtpredictor.shop/API/1_min.php/api/webapi"
API_RANDOM = "4a0522c6ecd8410496260e686be2a57c"
API_SIGNATURE = "334B5E70A0C9B8918B0B15E517E2069C"

predictions_history = []
last_api_time = 0
telegram_msg_ids = {}

# Game state variables (from HTML, adapted for Python)
last_period_number = None
history_data = [] # This will store resolved predictions for Bayesian adjustment and Loss Streak Strategy
current_result_prediction = None
pending_result = None
win_count = 0
loss_streak = 0
total_bets = 0
total_wins = 0
total_losses = 0
is_result_processing = False
last_timer_update = 0
cached_data = []
last_results = {'bigsmall': []} # Only bigsmall now

max_history_items_for_analysis = 100
full_history_fetch_interval = 30

# Engine performance tracking for auto-switching
engines = ['QuantumAI', 'NeuralNet', 'Fibonacci', 'Hybrid', 'ATS_Engine', 'RSI_Trend_Engine']
engine_performance = {}
for engine in engines:
    # Track performance per engine and per market regime
    engine_performance[engine] = {
        'overall': {'wins': 0, 'losses': 0, 'accuracy': 0, 'loss_streak': 0},
        'volatile': {'wins': 0, 'losses': 0, 'accuracy': 0, 'loss_streak': 0},
        'stable': {'wins': 0, 'losses': 0, 'accuracy': 0, 'loss_streak': 0},
        'trending': {'wins': 0, 'losses': 0, 'accuracy': 0, 'loss_streak': 0},
        'ranging': {'wins': 0, 'losses': 0, 'accuracy': 0, 'loss_streak': 0},
    }
current_engine = 'Hybrid'

# Q-Learning specific state
q_table = {}
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.1

# NeuralNet specific internal state for adaptive learning
# These weights simulate the learned parameters of a neural network.
# They will be adjusted based on performance.
neural_net_internal_weights = {
    'w_trend': {'value': 0.4, 'last_adj': 0.0},
    'w_parity': {'value': 0.3, 'last_adj': 0.0},
    'w_volatility': {'value': 0.2, 'last_adj': 0.0},
    'w_win_streak': {'value': 0.25, 'last_adj': 0.0},
    'w_loss_streak': {'value': -0.35, 'last_adj': 0.0},
    'w_time_remaining': {'value': 0.15, 'last_adj': 0.0},
    'w_pattern': {'value': 0.3, 'last_adj': 0.0},
    'bias': {'value': 0.1, 'last_adj': 0.0},
    # Weights for simulated hidden layer 1 (multiple nodes)
    'w_hidden_1_node_1_input_1': {'value': 0.2, 'last_adj': 0.0}, # f1_trend
    'w_hidden_1_node_1_input_2': {'value': -0.1, 'last_adj': 0.0}, # f3_volatility_norm
    'w_hidden_1_node_2_input_1': {'value': 0.15, 'last_adj': 0.0}, # f4_win_streak_norm
    'w_hidden_1_node_2_input_2': {'value': 0.25, 'last_adj': 0.0}, # f5_loss_streak_norm
    # Weights for simulated hidden layer 2 (connecting from HL1 outputs and other features)
    'w_hidden_2_node_1_input_1': {'value': 0.1, 'last_adj': 0.0}, # hidden_layer_output_1_node_1
    'w_hidden_2_node_1_input_2': {'value': -0.05, 'last_adj': 0.0}, # f6_time_remaining_norm
    'w_hidden_2_node_1_input_3': {'value': 0.08, 'last_adj': 0.0}, # f10_recurrent_avg_input_strength
    'w_hidden_2_node_1_input_4': {'value': -0.12, 'last_adj': 0.0}, # f12_recurrent_error_memory
    
    # Weights from hidden layers to output
    'w_output_from_hidden_1_node_1': {'value': 0.5, 'last_adj': 0.0}, 
    'w_output_from_hidden_1_node_2': {'value': 0.3, 'last_adj': 0.0}, 
    'w_output_from_hidden_2_node_1': {'value': 0.4, 'last_adj': 0.0}, 

    # Memory weights
    'w_prev_pred_strength': {'value': 0.1, 'last_adj': 0.0}, 
    'w_prev_actual_outcome': {'value': 0.15, 'last_adj': 0.0}, 
    'w_recurrent_avg_input_strength': {'value': 0.1, 'last_adj': 0.0}, 
    'w_recurrent_last_trend_val': {'value': 0.05, 'last_adj': 0.0}, 
    'w_recurrent_error_memory': {'value': -0.2, 'last_adj': 0.0}, 
    'w_recurrent_volatility_fingerprint': {'value': 0.05, 'last_adj': 0.0},
    'w_market_sentiment': {'value': 0.1, 'last_adj': 0.0},
    'w_trend_confirmation': {'value': 0.05, 'last_adj': 0.0},
    # New cross-feature interaction weights
    'w_trend_volatility_interaction': {'value': 0.05, 'last_adj': 0.0},
    'w_streak_time_interaction': {'value': 0.03, 'last_adj': 0.0},
    'w_parity_pattern_interaction': {'value': 0.04, 'last_adj': 0.0},
}

# Global constants for NN learning
NN_MOMENTUM_FACTOR = 0.9 # Factor for momentum in weight updates
NN_WEIGHT_DECAY = 0.0001 # Small decay to prevent weights from growing too large (L2 regularization)
NN_DROPOUT_RATE = 0.1 # Probability of dropping out an input feature (e.g., 10%)
NN_BATCH_SIZE = 5 # Number of predictions to accumulate before a weight update

# Storage for batch updates
nn_batch_inputs = []
nn_batch_errors = []

# Small temporary bias based on last high-confidence error, simulating short-term memory of mistakes
neural_net_last_error_bias = 0 
# New global states for NeuralNet's "memory"
neural_net_previous_prediction_strength = 0.0
neural_net_previous_actual_outcome = 'NONE' # Can be 'WIN', 'LOSS', or 'NONE'
# Recurrent state for more advanced memory (simplified RNN concept)
neural_net_recurrent_state = {
    'avg_input_strength': 0.0, 
    'last_trend_val': 0.0,
    'error_memory': 0.0, # Stores a decaying memory of past high-confidence errors
    'volatility_fingerprint': 0.0, # Stores a decaying memory of past volatility
    'long_term_error_avg': 0.0, # Longer term decaying average of errors
    'volatility_trend': 0.0 # Decaying average of volatility changes
}
# Feature importance scores (heuristic)
neural_net_feature_importance_scores = {
    'f1_trend': 0.1, 'f2_parity': 0.1, 'f3_volatility_norm': 0.1,
    'f4_win_streak_norm': 0.1, 'f5_loss_streak_norm': 0.1,
    'f6_time_remaining_norm': 0.1, 'f7_pattern_strength_norm': 0.1,
    'f8_prev_pred_strength': 0.1, 'f9_prev_actual_outcome_val': 0.1,
    'f10_recurrent_avg_input_strength': 0.1, 'f11_recurrent_last_trend_val': 0.1,
    'f12_recurrent_error_memory': 0.1, 'f13_recurrent_volatility_fingerprint': 0.1,
    'market_sentiment': 0.1, 'trend_confirmation': 0.1
}
# NeuralNet activation function parameters (learnable)
nn_activation_alpha = {'value': 1.0, 'last_adj': 0.0} # Controls steepness of tanh
nn_activation_beta = {'value': 1.0, 'last_adj': 0.0} # Controls range of tanh

# ==== Period Generator (from HTML) ====
def get_period():
    now = datetime.now(timezone.utc)
    year = now.year
    month = f"{now.month:02}"
    day = f"{now.day:02}"
    total_minutes = now.hour * 60 + now.minute
    period_num = f"{year}{month}{day}1000{10001 + total_minutes}"
    return period_num

def get_remaining_seconds():
    now = datetime.now(timezone.utc)
    return 60 - now.second

# Helper for Volatility (from HTML)
def calculate_volatility(numbers):
    if len(numbers) < 2: return 0
    # Calculate average absolute change over recent numbers (e.g., last 10)
    changes = [abs(numbers[i] - numbers[i - 1]) for i in range(1, min(10, len(numbers)))]
    avg_change = sum(changes) / max(1, len(changes))
    return avg_change

# Helper for Trend Direction (from HTML)
def get_trend_direction(numbers):
    if len(numbers) < 5: return 'Sideways'
    # Compare average of most recent 5 numbers to average of previous 5
    recent_avg = sum(numbers[:min(5, len(numbers))]) / min(5, len(numbers))
    older_avg = sum(numbers[5:min(10, len(numbers))]) / min(5, max(1, len(numbers) - 5))

    if recent_avg > older_avg * 1.1: return 'Up' # 10% increase
    if recent_avg < older_avg * 0.9: return 'Down' # 10% decrease
    return 'Sideways'

# New: Advanced Pattern Detector
def analyze_complex_sequence(historical_results, result_type, min_length=3, max_length=6, current_volatility=0):
    """
    Analyzes historical results for more complex, multi-period patterns.
    Adapts pattern search length based on volatility.
    Returns a dictionary with the detected pattern and its strength.
    """
    # Adjust max_length based on volatility: shorter patterns in high volatility
    effective_max_length = max_length
    if current_volatility > 3.0: # High volatility
        effective_max_length = min(max_length, 4)
    elif current_volatility < 1.0: # Low volatility
        effective_max_length = max(max_length, 6) # Longer patterns in low volatility (increased from original)

    relevant_results = [h for h in historical_results[result_type] if h is not None]
    if len(relevant_results) < min_length: return {'pattern': 'NONE', 'strength': 0, 'details': 'Not enough data', 'confidence': 0}

    patterns_found = {}

    for length in range(min_length, min(effective_max_length + 1, len(relevant_results) + 1)):
        current_slice = relevant_results[0:length]
        if len(current_slice) < length: continue

        # Alternating pattern (e.g., BIG, SMALL, BIG, SMALL)
        is_alternating = True
        for i in range(1, len(current_slice)):
            if current_slice[i] == current_slice[i - 1]:
                is_alternating = False
                break
        if is_alternating and len(current_slice) >= min_length:
            last_in_slice = current_slice[-1]
            predicted_next_alt = 'SMALL' if last_in_slice == 'BIG' else 'BIG' # Only bigsmall now

            patterns_found[f'ALT-{predicted_next_alt}'] = patterns_found.get(f'ALT-{predicted_next_alt}', 0) + 1

        # Repeating pattern (e.g., BIG, BIG, BIG)
        is_repeating = True
        for i in range(1, len(current_slice)):
            if current_slice[i] != current_slice[i - 1]:
                is_repeating = False
                break
        if is_repeating and len(current_slice) >= min_length:
            predicted_next_rep = current_slice[0]
            patterns_found[f'REP-{predicted_next_rep}'] = patterns_found.get(f'REP-{predicted_next_rep}', 0) + 1

        # Simple 3-period cycle (e.g., A-B-C-A-B-C)
        if length >= 3 and len(current_slice) >= 2 * length: # Need at least two cycles
            first_cycle = current_slice[0:length]
            second_cycle = current_slice[length:2*length]
            if first_cycle == second_cycle:
                predicted_next_cycle = first_cycle[0] # Predict the start of the next cycle
                patterns_found[f'CYCLE-{predicted_next_cycle}-{length}'] = patterns_found.get(f'CYCLE-{predicted_next_cycle}-{length}', 0) + 1.5 # Higher strength

    strongest_pattern_key = 'NONE'
    max_strength = 0
    predicted_outcome = 'NONE'
    pattern_details = 'No strong pattern'
    pattern_confidence = 0 # New: Confidence for the pattern itself

    for pattern_key, count in patterns_found.items():
        # Add recency bias: more recent patterns get a slight boost
        recency_bias_factor = 1.0
        # This is a simplification; in a real scenario, you'd track when each pattern was last seen.
        # For now, patterns found with shorter 'length' are implicitly more recent.
        if pattern_key.startswith(('ALT-', 'REP-')) and length <= 3:
            recency_bias_factor = 1.1
        elif pattern_key.startswith('CYCLE-') and int(pattern_key.split('-')[2]) <= 4:
            recency_bias_factor = 1.05
            
        adjusted_count = count * recency_bias_factor

        if adjusted_count > max_strength:
            max_strength = adjusted_count
            strongest_pattern_key = pattern_key
            
            if strongest_pattern_key.startswith('ALT-'):
                predicted_outcome = strongest_pattern_key.split('-')[1]
                pattern_details = f"Alternating pattern detected, predicting {predicted_outcome}."
                pattern_confidence = min(99, round(50 + (max_strength * 10))) # Heuristic confidence
            elif strongest_pattern_key.startswith('REP-'):
                predicted_outcome = strongest_pattern_key.split('-')[1]
                pattern_details = f"Repeating pattern detected, predicting {predicted_outcome}."
                pattern_confidence = min(99, round(50 + (max_strength * 10)))
            elif strongest_pattern_key.startswith('CYCLE-'):
                parts = strongest_pattern_key.split('-')
                predicted_outcome = parts[1]
                cycle_length = parts[2]
                pattern_details = f"Cycle of length {cycle_length} detected, predicting {predicted_outcome}."
                pattern_confidence = min(99, round(60 + (max_strength * 15))) # Cycles might imply higher confidence
                
    return {'pattern': predicted_outcome, 'strength': max_strength, 'details': pattern_details, 'confidence': pattern_confidence}


# Helper for Q-Learning state representation (enriched)
def get_q_state_improved(last_outcome, current_streak_magnitude, result_type, volatility_level, trend_direction, engine_performance_status):
    """
    Creates a detailed state representation for Q-learning.
    last_outcome: 'WIN', 'LOSS', 'NONE' (from previous prediction)
    current_streak_magnitude: absolute value of win/loss streak
    result_type: 'bigsmall'
    volatility_level: 'LOW', 'MEDIUM', 'HIGH'
    trend_direction: 'Up', 'Down', 'Sideways'
    engine_performance_status: 'WINNING', 'LOSING', 'NEUTRAL', 'ADAPTIVE'
    """
    return f"{last_outcome}-{ 'WIN' if current_streak_magnitude > 0 else ('LOSS' if current_streak_magnitude < 0 else 'NONE')}-{abs(current_streak_magnitude)}-{result_type}-{volatility_level}-{trend_direction}-{engine_performance_status}"

# Update Q-Table (simplified Q-Learning update)
def update_q_table(state, action, reward, next_state):
    """
    Updates the Q-table based on the observed reward and next state.
    This is a simplified Q-learning update for policy improvement.
    """
    if state not in q_table: q_table[state] = {}
    if action not in q_table[state]: q_table[state][action] = 0

    old_q = q_table[state][action]
    max_next_q = 0
    if next_state and next_state in q_table and q_table[next_state]:
        max_next_q = max(q_table[next_state].values())
    
    new_q = old_q + learning_rate * (reward + discount_factor * max_next_q - old_q)
    q_table[state][action] = new_q
    print(f"Q-Table Updated: State: {state}, Action: {action}, Old Q: {old_q:.2f}, New Q: {new_q:.2f}")

# Get action from Q-Table (simplified Q-Learning action selection)
def get_q_action(state):
    """
    Selects an action (engine) based on the current state using an epsilon-greedy policy.
    """
    if random.random() < exploration_rate or state not in q_table or not q_table[state]:
        # Explore: choose a random engine
        random_engine = random.choice(engines)
        print(f"Q-Learning: Exploring (random engine: {random_engine})")
        return random_engine
    else:
        # Exploit: choose the best action based on Q-values
        best_action = max(q_table[state], key=q_table[state].get)
        print(f"Q-Learning: Exploiting (best engine: {best_action}, Q: {q_table[state][best_action]:.2f})")
        return best_action

# Helper for Bayesian Adjustment (from HTML)
def bayesian_adjust(prediction, confidence, historical_data, result_type):
    """
    Adjusts confidence based on historical accuracy of similar predictions.
    Simulates a Bayesian update for confidence.
    """
    adjusted_confidence = confidence
    relevant_history = [h for h in historical_data if h['status'] in ['WIN 🎉', 'LOSS ❌']] # Check for overall status
    
    # Filter by resultType for specific prediction accuracy
    if result_type == 'bigsmall':
        relevant_history = [h for h in relevant_history if h['bigsmall_prediction'] == prediction]
        wins_for_this_prediction = len([h for h in relevant_history if h['bigsmall_prediction'] == prediction and h['bigsmall_result'] == h['bigsmall_prediction']])
    else: # Should only be bigsmall now, but keep for safety
        wins_for_this_prediction = 0 

    total_relevant = len(relevant_history)

    if total_relevant > 0:
        win_rate_for_this_prediction = wins_for_this_prediction / total_relevant
        # Adjust confidence more significantly if win rate for this prediction is very high or very low
        if win_rate_for_this_prediction > 0.6:
            adjusted_confidence += (win_rate_for_this_prediction - 0.6) * 20
        elif win_rate_for_this_prediction < 0.4:
            adjusted_confidence -= (0.4 - win_rate_for_this_prediction) * 20
    return min(99, max(0, round(adjusted_confidence)))

# Helper for Unexpected Result Identifying AI (from HTML)
def identify_unexpected_result(prediction, actual_result, confidence):
    """
    Logs unexpected results where high confidence prediction was wrong.
    This could be used for further analysis or model retraining in a real system.
    """
    global neural_net_last_error_bias
    if prediction != actual_result and confidence > 85:
        print(f"UNEXPECTED RESULT DETECTED: Predicted {prediction} with {confidence}% confidence, but actual was {actual_result}")
        # Apply a temporary error bias for NeuralNet if it was the one that made the high-confidence error
        if current_engine == 'NeuralNet':
            # If the prediction was 'BIG' and it was a LOSS, bias towards 'SMALL' next time
            if prediction == 'BIG' and actual_result == 'SMALL':
                neural_net_last_error_bias = -0.1 # Small negative bias for 'BIG'
            elif prediction == 'SMALL' and actual_result == 'BIG':
                neural_net_last_error_bias = 0.1 # Small positive bias for 'SMALL'
            print(f"NeuralNet: Applied temporary error bias of {neural_net_last_error_bias:.2f} due to high-confidence error.")
    else:
        neural_net_last_error_bias = 0 # Reset bias if prediction was correct or low confidence

# Generic fetch function with timeout and retries (from HTML)
def fetch_with_retry(url, json_data={}, retries=3, timeout=10):
    """
    Handles API calls with retries and timeouts for robustness.
    """
    for i in range(retries):
        try:
            response = requests.post(url, json=json_data, timeout=timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Fetch attempt {i + 1} for {url} failed: {e}")
            if i < retries - 1:
                time.sleep(2 * (i + 1)) # Exponential backoff
            else:
                raise # Re-raise exception after all retries fail

# Fetch game data from API (optimized for performance) (from HTML)
def fetch_optimized_data():
    """
    Fetches historical game data from the API, with caching to reduce API calls.
    """
    global cached_data, last_api_time
    now = time.time()
    # Only fetch if cache is old or not enough data
    if len(cached_data) >= max_history_items_for_analysis and (now - last_api_time < full_history_fetch_interval):
        return cached_data[:max_history_items_for_analysis]

    print("Fetching fresh historical data...")

    all_data = []
    # Fetch data in pages until max_history_items_for_analysis is met or no more data
    for i in range(1, (max_history_items_for_analysis // 10) + 1):
        try:
            page_data = fetch_page(i)
            if not page_data: # Stop if a page is empty
                break
            all_data.extend(page_data)
        except Exception as e:
            print(f"Error fetching page {i}: {e}")
            
    # Sort by issueNumber (period) in descending order and keep only the most recent
    cached_data = sorted([item for item in all_data if item and 'issueNumber' in item], 
                         key=lambda x: int(x['issueNumber']), reverse=True)[:max_history_items_for_analysis]

    last_api_time = now
    print(f"Fetched {len(cached_data)} historical records.")
    return cached_data

# New fetchPage function using the provided API (from HTML)
def fetch_page(page_no):
    """
    Fetches a single page of historical data from the game API.
    """
    request_data = {
        "pageSize": 10,
        "pageNo": page_no,
        "typeId": 1,
        "language": 0,
        "random": API_RANDOM,
        "signature": API_SIGNATURE,
        "timestamp": int(time.time()),
    }
    data = fetch_with_retry(f"{API_URL}/GetNoaverageEmerdList", json_data=request_data)
    return data.get("data", {}).get("list", [])

# Fetch single game result for current period check (from HTML)
def fetch_latest_actual_result(period):
    """
    Fetches the actual result for a specific period from the API.
    """
    request_data = {
        "pageSize": 10,
        "pageNo": 1, # Only need the latest page to find recent results
        "typeId": 1,
        "language": 0,
        "random": API_RANDOM,
        "signature": API_SIGNATURE,
        "timestamp": int(time.time()),
    }
    data = fetch_with_retry(f"{API_URL}/GetNoaverageEmerdList", json_data=request_data)
    
    list_data = data.get("data", {}).get("list", [])
    latest_result = next((item for item in list_data if item.get('issueNumber') == period), None)

    if latest_result:
        actual_number = int(latest_result['number']) % 10
        # Determine big/small
        actual_bigsmall = 'BIG' if actual_number >= 5 else 'SMALL'

        return {
            'period': latest_result['issueNumber'],
            'number_result': actual_number, # Keep number for volatility/trend calculation
            'bigsmall_result': actual_bigsmall,
        }
    return None

# --- Advanced Prediction Methods ---

def QuantumAIMethod():
    """
    QuantumAI Method: Uses statistical analysis (mean, std dev, momentum) and Fibonacci sequence detection.
    Returns predictions for big/small.
    """
    numbers = [int(item['number']) % 10 for item in cached_data] if cached_data else [random.randint(0, 9) for _ in range(20)]
    
    volatility = calculate_volatility(numbers)
    base_confidence = 85 - (volatility * 2)
    base_confidence = max(70, base_confidence)

    mean = sum(numbers[:min(20, len(numbers))]) / min(20, len(numbers))
    std_dev = (sum((x - mean)**2 for x in numbers[:min(20, len(numbers))]) / min(20, len(numbers)))**0.5
    upper_band = mean + 2 * std_dev
    lower_band = mean - 2 * std_dev
    latest = numbers[0]
    
    momentum = sum(numbers[i] - numbers[i + 1] for i in range(min(5, len(numbers)) - 1))

    fib_sequence = [0, 1, 1, 2, 3, 5, 8]
    fib_numbers_detected = len([n for n in numbers[:min(7, len(numbers))] if n in fib_sequence]) > 0

    results = {}
    levels = {}

    # Big/Small Prediction
    level1_bs = 'SMALL' if latest > upper_band else ('BIG' if latest < lower_band else ('BIG' if latest >= 5 else 'SMALL'))
    level1_bs_insight = f"Statistical range ({lower_band:.1f}-{upper_band:.1f}) suggests {level1_bs}."

    level2_bs = 'BIG' if momentum > 0 else 'SMALL'
    level2_bs_insight = f"Momentum ({momentum:.1f}) leans to {level2_bs}."

    level3_bs = 'BIG' if (latest % 2 == 0) else 'SMALL' if fib_numbers_detected else level1_bs # Simple parity check, or fib influence
    level3_bs_insight = f"Fibonacci numbers detected, using parity for {level3_bs}." if fib_numbers_detected else f"No direct Fib. numbers, using L1 for {level3_bs}."

    votes_bs = [level1_bs, level2_bs, level3_bs]
    result_bs = max(set(votes_bs), key=votes_bs.count)
    probability_bs = base_confidence + min(abs(latest - mean) / max(0.1, std_dev), 2) * 10
    
    # Counter-pattern for strong recent streaks
    if len(last_results['bigsmall']) >= 3:
        recent_actuals = last_results['bigsmall'][:3]
        if all(r == recent_actuals[0] for r in recent_actuals):
            last_actual = recent_actuals[0]
            counter_prediction = 'SMALL' if last_actual == 'BIG' else 'BIG'
            if result_bs == last_actual: # If our prediction is to continue the streak, consider reversing
                result_bs = counter_prediction
                probability_bs = max(0, probability_bs - 15) # Reduce confidence for reversal
                level3_bs_insight += f" (Reversed due to 3+ consecutive actual {last_actual} results)"
                print(f"QuantumAI: Reversed big/small prediction to {result_bs} due to actual streak of {last_actual}")

    results['bigsmall_result'] = result_bs
    levels['bigsmall'] = {'level1': level1_bs_insight, 'level2': level2_bs_insight, 'level3': level3_bs_insight}
    results['bigsmall_probability'] = min(99, max(0, round(probability_bs)))

    remaining_seconds = get_remaining_seconds()
    if remaining_seconds <= 10:
        results['bigsmall_probability'] = max(0, results['bigsmall_probability'] - 5)

    return {
        'bigsmall_result': results['bigsmall_result'], 'bigsmall_probability': results['bigsmall_probability'], 'bigsmall_levels': levels['bigsmall']
    }

def NeuralNetMethod():
    """
    NeuralNet Method: Simulates an advanced neural network with multi-layered feature processing,
    adaptive confidence, and sophisticated indicator analysis.
    Returns predictions for big/small.
    """
    global neural_net_internal_weights, neural_net_last_error_bias, \
           neural_net_previous_prediction_strength, neural_net_previous_actual_outcome, \
           neural_net_recurrent_state, neural_net_feature_importance_scores, \
           nn_activation_alpha, nn_activation_beta

    numbers = [int(item['number']) % 10 for item in cached_data] if cached_data else [random.randint(0, 9) for _ in range(20)]
    if not numbers:
        return {
            'bigsmall_result': "N/A", 'bigsmall_probability': 0, 'bigsmall_levels': {'level1': 'No data', 'level2': 'No data', 'level3': 'No data'},
            'nn_input_features': {}
        }

    # --- Feature Engineering & Layer 1 (Input Processing) ---
    current_volatility = calculate_volatility(numbers)
    trend_direction = get_trend_direction(numbers)
    
    # Adaptive base confidence based on NeuralNet's own performance
    nn_accuracy = engine_performance['NeuralNet']['overall']['accuracy'] # Use overall accuracy for base confidence
    nn_loss_streak = engine_performance['NeuralNet']['overall']['loss_streak']
    
    base_confidence = 90 - (current_volatility * 2.5) # Stronger penalty for volatility
    
    # Dynamic internal learning rate for NeuralNet (simulated self-tuning)
    # If accuracy is low, increase learning rate to adapt faster. If high, decrease to stabilize.
    nn_learning_rate = 0.05 # Base learning rate
    if nn_accuracy < 0.5:
        nn_learning_rate += (0.5 - nn_accuracy) * 0.1 # Increase learning rate if accuracy is below 50%
    elif nn_accuracy > 0.7:
        nn_learning_rate -= (nn_accuracy - 0.7) * 0.05 # Decrease learning rate if accuracy is very high
    nn_learning_rate = max(0.01, min(0.2, nn_learning_rate)) # Clamp between 0.01 and 0.2

    # Adjust base confidence based on NN's self-performance (simulated learning)
    if nn_accuracy > 0.6:
        base_confidence += (nn_accuracy - 0.6) * 15 # Boost confidence if NN is doing well
    if nn_loss_streak >= 2:
        base_confidence -= nn_loss_streak * 7 # Reduce confidence more aggressively on loss streak
    
    base_confidence = max(60, min(95, base_confidence)) # Clamp confidence within a reasonable range

    # Advanced RSI Calculation (Wilder's smoothing)
    gains = [0] * len(numbers)
    losses = [0] * len(numbers)
    for i in range(len(numbers) - 1):
        change = numbers[i] - numbers[i+1]
        if change > 0: gains[i] = change
        else: losses[i] = abs(change)

    rsi_period = min(len(numbers) - 1, 14) 
    avg_gain_smooth = sum(gains[:rsi_period]) / max(1e-10, rsi_period) if rsi_period > 0 else 0
    avg_loss_smooth = sum(losses[:rsi_period]) / max(1e-10, rsi_period) if rsi_period > 0 else 0

    for i in range(rsi_period, len(numbers) - 1):
        avg_gain_smooth = (avg_gain_smooth * (rsi_period - 1) + gains[i]) / rsi_period
        avg_loss_smooth = (avg_loss_smooth * (rsi_period - 1) + losses[i]) / rsi_period

    rs = avg_gain_smooth / max(1e-10, avg_loss_smooth)
    rsi = 100 - (100 / (1 + rs)) if rs else 50
    level1_insight = f"Advanced RSI ({rsi:.2f}) value. "

    # Triple Exponential Moving Average (TEMA)
    def ema(data_series, period):
        if not data_series: return []
        k = 2 / (period + 1)
        ema_values = [data_series[0]]
        for i in range(1, len(data_series)):
            ema_val = (data_series[i] * k) + (ema_values[-1] * (1 - k))
            ema_values.append(ema_val)
        return ema_values

    tema_period = 10 
    ema1_series = ema(numbers, tema_period)
    if len(ema1_series) < tema_period: 
        return {
            'bigsmall_result': "N/A", 'bigsmall_probability': 0, 'bigsmall_levels': {'level1': 'Not enough data for TEMA', 'level2': '', 'level3': ''},
            'nn_input_features': {}
        }
    
    ema2_series = ema(ema1_series, tema_period)
    if len(ema2_series) < tema_period: 
        return {
            'bigsmall_result': "N/A", 'bigsmall_probability': 0, 'bigsmall_levels': {'level1': 'Not enough data for TEMA', 'level2': '', 'level3': ''},
            'nn_input_features': {}
        }

    ema3_series = ema(ema2_series, tema_period)
    if len(ema3_series) < tema_period: 
        return {
            'bigsmall_result': "N/A", 'bigsmall_probability': 0, 'bigsmall_levels': {'level1': 'Not enough data for TEMA', 'level2': '', 'level3': ''},
            'nn_input_features': {}
        }

    current_tema = (3 * ema1_series[-1]) - (3 * ema2_series[-1]) + ema3_series[-1]
    
    ma_bias = numbers[0] - current_tema if numbers and current_tema != 0 else 0
    level2_insight = f"TEMA ({current_tema:.2f}) bias: {ma_bias:.2f}. "

    # --- Layer 2 (Complex Feature Interaction & Pattern Recognition) ---
    # Input features for simulated "neuron"
    f1_trend = 1 if trend_direction == 'Up' else (-1 if trend_direction == 'Down' else 0)
    f2_parity = 1 if numbers[0] % 2 == 0 else -1 if numbers else 0
    f3_volatility_norm = min(1, current_volatility / 5.0) 

    f4_win_streak_norm = min(1, win_count / 10.0) # Normalize over a larger potential streak
    f5_loss_streak_norm = min(1, loss_streak / 10.0) 

    f6_time_remaining_norm = get_remaining_seconds() / 60.0 # Normalize seconds remaining (0-1)

    # Big/Small Pattern
    detected_pattern_bs_info = analyze_complex_sequence(last_results, 'bigsmall', current_volatility=current_volatility)
    f7_pattern_strength_bs_norm = detected_pattern_bs_info['strength'] / 6.0
    f7_pattern_details_bs = detected_pattern_bs_info['details']
    f7_pattern_confidence_bs_norm = detected_pattern_bs_info['confidence'] / 100.0

    # New memory features
    f8_prev_pred_strength = neural_net_previous_prediction_strength # Previous NN prediction strength
    f9_prev_actual_outcome_val = 1 if neural_net_previous_actual_outcome == 'WIN' else (-1 if neural_net_previous_actual_outcome == 'LOSS' else 0)

    # Recurrent state input
    f10_recurrent_avg_input_strength = neural_net_recurrent_state['avg_input_strength']
    f11_recurrent_last_trend_val = neural_net_recurrent_state['last_trend_val']
    f12_recurrent_error_memory = neural_net_recurrent_state['error_memory']
    f13_recurrent_volatility_fingerprint = neural_net_recurrent_state['volatility_fingerprint']
    f14_long_term_error_avg = neural_net_recurrent_state['long_term_error_avg']
    f15_volatility_trend = neural_net_recurrent_state['volatility_trend']


    # Store input features for potential backpropagation
    nn_input_features = {
        'f1_trend': f1_trend, 'f2_parity': f2_parity, 'f3_volatility_norm': f3_volatility_norm,
        'f4_win_streak_norm': f4_win_streak_norm, 'f5_loss_streak_norm': f5_loss_streak_norm,
        'f6_time_remaining_norm': f6_time_remaining_norm, 
        'f7_pattern_strength_bs_norm': f7_pattern_strength_bs_norm,
        'f7_pattern_confidence_bs_norm': f7_pattern_confidence_bs_norm,
        'f8_prev_pred_strength': f8_prev_pred_strength, 'f9_prev_actual_outcome_val': f9_prev_actual_outcome_val,
        'f10_recurrent_avg_input_strength': f10_recurrent_avg_input_strength,
        'f11_recurrent_last_trend_val': f11_recurrent_last_trend_val,
        'f12_recurrent_error_memory': f12_recurrent_error_memory,
        'f13_recurrent_volatility_fingerprint': f13_recurrent_volatility_fingerprint,
        'f14_long_term_error_avg': f14_long_term_error_avg,
        'f15_volatility_trend': f15_volatility_trend,
        'current_tema': current_tema, 'ma_bias': ma_bias, 'rsi': rsi
    }

    # Simulated Dropout: Randomly set some input features to 0
    active_features = {}
    for feature_name, value in nn_input_features.items():
        if random.random() > NN_DROPOUT_RATE: # Keep feature with (1 - dropout_rate) probability
            active_features[feature_name] = value
        else:
            active_features[feature_name] = 0.0 # Drop feature

    # Simulated Hidden Layer 1 calculation (multiple nodes)
    # Activation function with adaptive slope (nn_activation_beta)
    hidden_layer_input_1_node_1 = (active_features.get('f1_trend', 0.0) * neural_net_internal_weights['w_hidden_1_node_1_input_1']['value']) + \
                                  (active_features.get('f3_volatility_norm', 0.0) * neural_net_internal_weights['w_hidden_1_node_1_input_2']['value'])
    hidden_layer_output_1_node_1 = nn_activation_alpha['value'] * math.tanh(nn_activation_beta['value'] * hidden_layer_input_1_node_1) 

    hidden_layer_input_1_node_2 = (active_features.get('f4_win_streak_norm', 0.0) * neural_net_internal_weights['w_hidden_1_node_2_input_1']['value']) + \
                                  (active_features.get('f5_loss_streak_norm', 0.0) * neural_net_internal_weights['w_hidden_1_node_2_input_2']['value'])
    hidden_layer_output_1_node_2 = nn_activation_alpha['value'] * math.tanh(nn_activation_beta['value'] * hidden_layer_input_1_node_2) 

    # Simulated Hidden Layer 2 calculation (more complex interactions)
    hidden_layer_input_2_node_1 = (hidden_layer_output_1_node_1 * neural_net_internal_weights['w_hidden_2_node_1_input_1']['value']) + \
                                  (active_features.get('f6_time_remaining_norm', 0.0) * neural_net_internal_weights['w_hidden_2_node_1_input_2']['value']) + \
                                  (active_features.get('f10_recurrent_avg_input_strength', 0.0) * neural_net_internal_weights['w_hidden_2_node_1_input_3']['value']) + \
                                  (active_features.get('f12_recurrent_error_memory', 0.0) * neural_net_internal_weights['w_hidden_2_node_1_input_4']['value'])
    hidden_layer_output_2_node_1 = nn_activation_alpha['value'] * math.tanh(nn_activation_beta['value'] * hidden_layer_input_2_node_1)

    # Complex raw activation combining features with non-linear terms (simulating deeper network layers)
    # Inputs are scaled by feature importance scores
    raw_activation = (active_features.get('f1_trend', 0.0) * neural_net_internal_weights['w_trend']['value'] * neural_net_feature_importance_scores['f1_trend']) + \
                     (active_features.get('f2_parity', 0.0) * neural_net_internal_weights['w_parity']['value'] * neural_net_feature_importance_scores['f2_parity']) + \
                     (active_features.get('f3_volatility_norm', 0.0) * neural_net_internal_weights['w_volatility']['value'] * neural_net_feature_importance_scores['f3_volatility_norm']) + \
                     (active_features.get('f4_win_streak_norm', 0.0) * neural_net_internal_weights['w_win_streak']['value'] * neural_net_feature_importance_scores['f4_win_streak_norm']) + \
                     (active_features.get('f5_loss_streak_norm', 0.0) * neural_net_internal_weights['w_loss_streak']['value'] * neural_net_feature_importance_scores['f5_loss_streak_norm']) + \
                     (active_features.get('f6_time_remaining_norm', 0.0) * neural_net_internal_weights['w_time_remaining']['value'] * neural_net_feature_importance_scores['f6_time_remaining_norm']) + \
                     (active_features.get('f7_pattern_strength_bs_norm', 0.0) * neural_net_internal_weights['w_pattern']['value'] * neural_net_feature_importance_scores['f7_pattern_strength_norm']) + \
                     (active_features.get('f8_prev_pred_strength', 0.0) * neural_net_internal_weights['w_prev_pred_strength']['value'] * neural_net_feature_importance_scores['f8_prev_pred_strength']) + \
                     (active_features.get('f9_prev_actual_outcome_val', 0.0) * neural_net_internal_weights['w_prev_actual_outcome']['value'] * neural_net_feature_importance_scores['f9_prev_actual_outcome_val']) + \
                     (active_features.get('f10_recurrent_avg_input_strength', 0.0) * neural_net_internal_weights['w_recurrent_avg_input_strength']['value'] * neural_net_feature_importance_scores['f10_recurrent_avg_input_strength']) + \
                     (active_features.get('f11_recurrent_last_trend_val', 0.0) * neural_net_internal_weights['w_recurrent_last_trend_val']['value'] * neural_net_feature_importance_scores['f11_recurrent_last_trend_val']) + \
                     (active_features.get('f12_recurrent_error_memory', 0.0) * neural_net_internal_weights['w_recurrent_error_memory']['value'] * neural_net_feature_importance_scores['f12_recurrent_error_memory']) + \
                     (active_features.get('f13_recurrent_volatility_fingerprint', 0.0) * neural_net_internal_weights['w_recurrent_volatility_fingerprint']['value'] * neural_net_feature_importance_scores['f13_recurrent_volatility_fingerprint']) + \
                     neural_net_internal_weights['bias']['value'] + \
                     neural_net_last_error_bias # Incorporate temporary error bias

    # Add more non-linear interactions (simulating deeper network layers)
    # These terms also incorporate feature importance
    raw_activation += (active_features.get('f1_trend', 0.0) * active_features.get('f3_volatility_norm', 0.0) * neural_net_internal_weights['w_trend_volatility_interaction']['value'] * neural_net_feature_importance_scores['f1_trend'] * neural_net_feature_importance_scores['f3_volatility_norm'] * (1 + nn_learning_rate)) 
    raw_activation += (active_features.get('f4_win_streak_norm', 0.0) * active_features.get('f6_time_remaining_norm', 0.0) * neural_net_internal_weights['w_streak_time_interaction']['value'] * neural_net_feature_importance_scores['f4_win_streak_norm'] * neural_net_feature_importance_scores['f6_time_remaining_norm'] * (1 + nn_learning_rate)) 
    raw_activation += (active_features.get('f2_parity', 0.0) * active_features.get('f7_pattern_strength_bs_norm', 0.0) * neural_net_internal_weights['w_parity_pattern_interaction']['value'] * neural_net_feature_importance_scores['f2_parity'] * neural_net_feature_importance_scores['f7_pattern_strength_norm'] * (1 + nn_learning_rate)) 

    # Incorporate the simulated hidden layer output into the main activation
    raw_activation += (hidden_layer_output_1_node_1 * neural_net_internal_weights['w_output_from_hidden_1_node_1']['value'])
    raw_activation += (hidden_layer_output_1_node_2 * neural_net_internal_weights['w_output_from_hidden_1_node_2']['value']) 
    raw_activation += (hidden_layer_output_2_node_1 * neural_net_internal_weights['w_output_from_hidden_2_node_1']['value']) 

    # Simulated contextual feature generation
    market_sentiment_score = (active_features.get('rsi', 50) - 50) / 50.0 + active_features.get('ma_bias', 0.0) / 5.0 # Combines RSI and TEMA bias
    trend_confirmation = active_features.get('f1_trend', 0.0) * active_features.get('f3_volatility_norm', 0.0) # Stronger trend in low volatility
    
    raw_activation += (market_sentiment_score * neural_net_internal_weights['w_market_sentiment']['value'] * neural_net_feature_importance_scores['market_sentiment'])
    raw_activation += (trend_confirmation * neural_net_internal_weights['w_trend_confirmation']['value'] * neural_net_feature_importance_scores['trend_confirmation'])
    
    # Sigmoid-like activation to get a "prediction bias" between -1 and 1
    # Using tanh for a smooth output range
    prediction_strength = nn_activation_alpha['value'] * math.tanh(nn_activation_beta['value'] * raw_activation / 2.0) 
    level3_insight = (
        f"Complex pattern activation ({prediction_strength:.2f}) from features, adaptive weights (NN Learn Rate: {nn_learning_rate:.3f}), "
        f"multi-layered hidden processing, recurrent memory, and contextual features. Pattern: {f7_pattern_details_bs}"
    )

    # --- Layer 3 (Output & Decision Layer) ---
    results = {}
    levels = {}

    # Big/Small Prediction
    vote_rsi_bs = ''
    if rsi > 70: vote_rsi_bs = 'SMALL' # Overbought -> reversal (small)
    elif rsi < 30: vote_rsi_bs = 'BIG' # Oversold -> reversal (big)
    else: vote_rsi_bs = 'BIG' if numbers[0] >= 5 else 'SMALL' # Neutral RSI, use current number parity

    vote_tema_bs = 'BIG' if ma_bias > 0 else 'SMALL'

    # Directly use prediction_strength for the pattern vote
    vote_pattern_bs = 'BIG' if prediction_strength > 0 else 'SMALL'

    vote_counts_bs = {'BIG': 0, 'SMALL': 0}

    accuracy_influence_factor = 1 + (nn_accuracy - 0.5)
    vote_counts_bs[vote_rsi_bs] = vote_counts_bs.get(vote_rsi_bs, 0) + (1.8 * accuracy_influence_factor) 
    vote_counts_bs[vote_tema_bs] = vote_counts_bs.get(vote_tema_bs, 0) + (1.5 * accuracy_influence_factor)
    vote_counts_bs[vote_pattern_bs] = vote_counts_bs.get(vote_pattern_bs, 0) + (1.3 * accuracy_influence_factor)

    result_bs = max(vote_counts_bs, key=vote_counts_bs.get)
    probability_bs = (prediction_strength + 1) / 2 * 100
    probability_bs = base_confidence * (probability_bs / 100.0)
    probability_bs = max(0, min(99, round(probability_bs)))
    probability_bs = bayesian_adjust(result_bs, probability_bs, history_data, 'bigsmall')

    if len(last_results['bigsmall']) >= 3:
        recent_actuals = last_results['bigsmall'][:3]
        if all(r == recent_actuals[0] for r in recent_actuals):
            last_actual = recent_actuals[0]
            counter_prediction = 'SMALL' if last_actual == 'BIG' else 'BIG'
            if result_bs == last_actual:
                result_bs = counter_prediction
                probability_bs = max(0, probability_bs - 20)
                level3_insight += f" (NN reversed big/small due to strong actual {last_actual} streak)"
                print(f"NeuralNet: Reversed big/small prediction to {result_bs} due to actual streak of {last_actual}")

    results['bigsmall_result'] = result_bs
    results['bigsmall_probability'] = probability_bs
    levels['bigsmall'] = {'level1': level1_insight, 'level2': level2_insight, 'level3': level3_insight}

    # Add a slight penalty if prediction is made very close to period end
    remaining_seconds = get_remaining_seconds()
    if remaining_seconds <= 5: # Very last few seconds
        results['bigsmall_probability'] = max(0, results['bigsmall_probability'] - 10)
        level3_insight += " (Time sensitivity applied - very late prediction)"
    elif remaining_seconds <= 15:
        results['bigsmall_probability'] = max(0, results['bigsmall_probability'] - 3)
        level3_insight += " (Time sensitivity applied)"

    return {
        'bigsmall_result': results['bigsmall_result'], 'bigsmall_probability': results['bigsmall_probability'], 'bigsmall_levels': levels['bigsmall'],
        'nn_input_features': nn_input_features, # Return input features for backprop
        'nn_prediction_strength': prediction_strength # Return raw prediction strength
    }

def FibonacciMethod():
    """
    Fibonacci Method: Uses Fibonacci retracement and time zone analysis.
    Returns predictions for big/small.
    """
    numbers = [int(item['number']) % 10 for item in cached_data] if cached_data else [random.randint(0, 9) for _ in range(20)]

    volatility = calculate_volatility(numbers)
    base_confidence = 85 - (volatility * 2)
    base_confidence = max(70, base_confidence)

    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    relevant_numbers_for_fib = numbers[:min(20, len(numbers))]
    if len(relevant_numbers_for_fib) < 2: 
        return {
            'bigsmall_result': "N/A", 'bigsmall_probability': 0, 'bigsmall_levels': {'level1': 'Not enough data', 'level2': 'Not enough data', 'level3': 'Not enough data'}
        }

    high = max(relevant_numbers_for_fib)
    low = min(relevant_numbers_for_fib)
    range_val = high - low
    current = numbers[0]

    closest_fib = current
    if range_val > 0:
        fib_levels = [low + range_val * r for r in fib_ratios]
        closest_fib = min(fib_levels, key=lambda x: abs(x - current))

    time_zones = [5, 8, 13, 21] # Fibonacci time zones in minutes/periods
    current_zone_index = -1
    for i, tz in enumerate(time_zones):
        if len(numbers) >= tz: # Check if enough historical periods for this time zone
            current_zone_index = i
    zone_factor = (current_zone_index + 1) / len(time_zones) if current_zone_index >= 0 else 0.5

    extension_levels = [1.272, 1.414, 1.618]
    extension_score = len([e for e in extension_levels if current > (low + range_val * e)])

    results = {}
    levels = {}

    # Big/Small Prediction
    level1_bs = 'BIG' if current > closest_fib else 'SMALL' # Prediction based on current number vs closest fib level
    level1_bs_insight = f"Fib. retracement near {closest_fib:.1f} suggests {level1_bs}."

    level2_bs = 'BIG' if zone_factor > 0.5 else 'SMALL' # Influence from Fibonacci time zones
    level2_bs_insight = f"Fib. time zone aligns with {level2_bs}."

    level3_bs = 'BIG' if extension_score > 0 else 'SMALL' # Influence from Fibonacci extensions
    level3_bs_insight = f"Fib. extension count {extension_score} suggests {level3_bs}."

    votes_bs = [level1_bs, level2_bs, level3_bs]
    result_bs = max(set(votes_bs), key=votes_bs.count)
    probability_bs = base_confidence + (abs(current - closest_fib) / max(0.1, range_val)) * 50 if range_val > 0 else 0

    if len(last_results['bigsmall']) >= 3:
        recent_actuals = last_results['bigsmall'][:3]
        if all(r == recent_actuals[0] for r in recent_actuals):
            last_actual = recent_actuals[0]
            counter_prediction = 'SMALL' if last_actual == 'BIG' else 'BIG'
            if result_bs == last_actual:
                result_bs = counter_prediction
                probability_bs = max(0, probability_bs - 15)
                level3_bs_insight += f" (Reversed due to 3+ consecutive actual {last_actual} results)"
                print(f"Fibonacci: Reversed big/small prediction to {result_bs} due to actual streak of {last_actual}")

    results['bigsmall_result'] = result_bs
    results['bigsmall_probability'] = min(99, max(0, round(probability_bs)))
    levels['bigsmall'] = {'level1': level1_bs_insight, 'level2': level2_bs_insight, 'level3': level3_bs_insight}

    remaining_seconds = get_remaining_seconds()
    if remaining_seconds <= 10:
        results['bigsmall_probability'] = max(0, results['bigsmall_probability'] - 5)

    return {
        'bigsmall_result': results['bigsmall_result'], 'bigsmall_probability': results['bigsmall_probability'], 'bigsmall_levels': levels['bigsmall']
    }

def HybridMethod():
    """
    Hybrid Method: Combines predictions from QuantumAI, NeuralNet, and Fibonacci methods
    with weighted voting and Q-learning influence.
    Returns predictions for big/small.
    """
    quantum = QuantumAIMethod()
    neural = NeuralNetMethod() # Get full neural output
    fib = FibonacciMethod()

    results = {}
    levels = {}

    data_numbers = [int(item['number']) % 10 for item in cached_data] if cached_data else []
    current_volatility = calculate_volatility(data_numbers)
    volatility_level = 'MEDIUM'
    if current_volatility < 1.5: volatility_level = 'LOW'
    elif current_volatility > 3: volatility_level = 'HIGH'

    current_trend = get_trend_direction(data_numbers)

    engine_status = 'NEUTRAL'
    if current_engine in engine_performance:
        if engine_performance[current_engine]['overall']['accuracy'] > 0.6: engine_status = 'WINNING'
        elif engine_performance[current_engine]['overall']['loss_streak'] >= 2: engine_status = 'LOSING'

    # Dynamic weighting based on engine performance and market conditions
    base_quantum_weight = 1.5
    base_neural_weight = 1.2
    base_fib_weight = 1.0

    dynamic_quantum_weight = base_quantum_weight * (1 + (engine_performance['QuantumAI']['overall']['accuracy'] - 0.5) * 0.5)
    dynamic_neural_weight = base_neural_weight * (1 + (engine_performance['NeuralNet']['overall']['accuracy'] - 0.5) * 0.7)
    dynamic_fib_weight = base_fib_weight * (1 + (engine_performance['Fibonacci']['overall']['accuracy'] - 0.5) * 0.4)

    market_regime = "Neutral"
    if current_volatility > 3.0: market_regime = "Volatile"
    elif current_volatility < 1.0: market_regime = "Stable"
    
    if market_regime == "Volatile":
        dynamic_neural_weight *= 1.2
        dynamic_quantum_weight *= 0.9
    elif market_regime == "Stable":
        dynamic_fib_weight *= 1.1
        dynamic_neural_weight *= 0.95
    
    if current_trend == 'Up' or current_trend == 'Down':
        dynamic_quantum_weight *= 1.05
        dynamic_fib_weight *= 1.05
    else:
        dynamic_neural_weight *= 1.05

    # Q-learning action influences the voting
    current_q_state = get_q_state_improved(predictions_history[0]['status'] if predictions_history else 'NONE', win_count if win_count > 0 else -loss_streak, 'bigsmall', volatility_level, current_trend, engine_status)
    q_action = get_q_action(current_q_state)

    # --- Big/Small Prediction ---
    votes_bs = {}
    votes_bs[quantum['bigsmall_result']] = votes_bs.get(quantum['bigsmall_result'], 0) + dynamic_quantum_weight
    votes_bs[neural['bigsmall_result']] = votes_bs.get(neural['bigsmall_result'], 0) + dynamic_neural_weight
    votes_bs[fib['bigsmall_result']] = votes_bs.get(fib['bigsmall_result'], 0) + dynamic_fib_weight

    if q_action == 'QuantumAI': votes_bs[quantum['bigsmall_result']] = votes_bs.get(quantum['bigsmall_result'], 0) + 0.5
    if q_action == 'NeuralNet': votes_bs[neural['bigsmall_result']] = votes_bs.get(neural['bigsmall_result'], 0) + 0.5
    if q_action == 'Fibonacci': votes_bs[fib['bigsmall_result']] = votes_bs.get(fib['bigsmall_result'], 0) + 0.5
    if q_action == 'Hybrid':
        votes_bs[quantum['bigsmall_result']] = votes_bs.get(quantum['bigsmall_result'], 0) + 0.2
        votes_bs[neural['bigsmall_result']] = votes_bs.get(neural['bigsmall_result'], 0) + 0.2
        votes_bs[fib['bigsmall_result']] = votes_bs.get(fib['bigsmall_result'], 0) + 0.2
    
    result_bs = max(votes_bs, key=votes_bs.get)
    total_dynamic_weight = dynamic_quantum_weight + dynamic_neural_weight + dynamic_fib_weight
    probability_bs = (quantum['bigsmall_probability'] * dynamic_quantum_weight + 
                      neural['bigsmall_probability'] * dynamic_neural_weight + 
                      fib['bigsmall_probability'] * dynamic_fib_weight) / max(1e-10, total_dynamic_weight)
    probability_bs = bayesian_adjust(result_bs, probability_bs, history_data, 'bigsmall')

    if len(last_results['bigsmall']) >= 3:
        recent_actuals = last_results['bigsmall'][:3]
        if all(r == recent_actuals[0] for r in recent_actuals):
            last_actual = recent_actuals[0]
            counter_prediction = 'SMALL' if last_actual == 'BIG' else 'BIG'
            if result_bs == last_actual:
                result_bs = counter_prediction
                probability_bs = max(0, probability_bs - 15)
                print(f"Hybrid: Reversed big/small prediction to {result_bs} due to actual streak of {last_actual}")

    results['bigsmall_result'] = result_bs
    results['bigsmall_probability'] = min(99, max(0, round(probability_bs)))
    levels['bigsmall'] = {
        'level1': f"QAI: {quantum['bigsmall_levels']['level1']} | NN: {neural['bigsmall_levels']['level1']} | Fib: {fib['bigsmall_levels']['level1']}",
        'level2': f"QAI: {quantum['bigsmall_levels']['level2']} | NN: {neural['bigsmall_levels']['level2']} | Fib: {fib['bigsmall_levels']['level2']}",
        'level3': f"QAI: {quantum['bigsmall_levels']['level3']} | NN: {neural['bigsmall_levels']['level3']} | Fib: {fib['bigsmall_levels']['level3']}"
    }

    return {
        'bigsmall_result': results['bigsmall_result'], 'bigsmall_probability': results['bigsmall_probability'], 'bigsmall_levels': levels['bigsmall'],
        'nn_input_features': neural.get('nn_input_features', {}), # Pass NN features through Hybrid
        'nn_prediction_strength': neural.get('nn_prediction_strength', 0.0) # Pass NN raw strength through Hybrid
    }

def ATS_Engine():
    """
    ATS (Automated Trading System) Engine: Uses momentum, ARIMA-like prediction, and sophisticated pattern analysis.
    Returns predictions for big/small.
    """
    numbers = [int(item['number']) % 10 for item in cached_data] if cached_data else [random.randint(0, 9) for _ in range(20)]

    volatility = calculate_volatility(numbers)
    base_confidence = 88 - (volatility * 3)
    base_confidence = max(75, base_confidence)

    momentum = sum(numbers[i] - numbers[i + 1] for i in range(min(10, len(numbers)) - 1))
    level1_insight = f"Recent momentum: {momentum:.1f}."

    arima_prediction_value = numbers[0]
    if len(numbers) >= 2:
        diff = numbers[0] - numbers[1]
        arima_prediction_value = numbers[0] + diff # Simple linear extrapolation
    level2_insight = f"ARIMA-like model predicts: {arima_prediction_value:.1f}."

    results = {}
    levels = {}

    # Big/Small Prediction
    detected_pattern_bs_info = analyze_complex_sequence(last_results, 'bigsmall', current_volatility=volatility)
    detected_pattern_bs = detected_pattern_bs_info['pattern']
    pattern_strength_bs = detected_pattern_bs_info['strength']
    pattern_influence_bs = 0
    if detected_pattern_bs != 'NONE' and pattern_strength_bs > 0:
        pattern_influence_bs = 10
        level3_bs_insight = f"Sophisticated pattern '{detected_pattern_bs}' (Str: {pattern_strength_bs}) detected."
    else:
        level3_bs_insight = 'No strong sophisticated pattern detected.'

    pred_bs = 'BIG' if arima_prediction_value >= 5 else 'SMALL'
    if momentum > 0 and pred_bs == 'SMALL': pred_bs = 'BIG'
    if momentum < 0 and pred_bs == 'BIG': pred_bs = 'SMALL'
    if detected_pattern_bs != 'NONE' and pattern_influence_bs > 0: pred_bs = detected_pattern_bs

    result_bs = pred_bs
    probability_bs = base_confidence + abs(momentum) * 2 + pattern_influence_bs
    probability_bs = bayesian_adjust(result_bs, probability_bs, history_data, 'bigsmall')

    if len(last_results['bigsmall']) >= 3:
        recent_actuals = last_results['bigsmall'][:3]
        if all(r == recent_actuals[0] for r in recent_actuals):
            last_actual = recent_actuals[0]
            counter_prediction = 'SMALL' if last_actual == 'BIG' else 'BIG'
            if result_bs == last_actual:
                result_bs = counter_prediction
                probability_bs = max(0, probability_bs - 15)
                level3_bs_insight += f" (Reversed due to 3+ consecutive actual {last_actual} results)"
                print(f"ATS_Engine: Reversed big/small prediction to {result_bs} due to actual streak of {last_actual}")

    results['bigsmall_result'] = result_bs
    results['bigsmall_probability'] = min(99, max(0, round(probability_bs)))
    levels['bigsmall'] = {'level1': level1_insight, 'level2': level2_insight, 'level3': level3_bs_insight}

    return {
        'bigsmall_result': results['bigsmall_result'], 'bigsmall_probability': results['bigsmall_probability'], 'bigsmall_levels': levels['bigsmall']
    }

def RSI_Trend_Engine():
    """
    RSI Trend Engine: Focuses on Relative Strength Index and overall trend direction.
    Returns predictions for big/small.
    """
    numbers = [int(item['number']) % 10 for item in cached_data] if cached_data else [random.randint(0, 9) for _ in range(20)]

    volatility = calculate_volatility(numbers)
    base_confidence = 86 - (volatility * 2.5)
    base_confidence = max(72, base_confidence)

    avg_gain = 0
    avg_loss = 0
    rsi_period = min(len(numbers) - 1, 14)
    if rsi_period > 0:
        for i in range(rsi_period):
            change = numbers[i] - numbers[i + 1]
            if change > 0: avg_gain += change
            else: avg_loss += abs(change)
    
    rs = avg_gain / max(1e-10, avg_loss) if avg_loss != 0 or avg_gain != 0 else 1 # Handle division by zero
    rsi = 100 - (100 / (1 + rs)) if rs else 50
    level1_insight = f"RSI ({rsi:.2f}) value."

    trend = get_trend_direction(numbers)
    level2_insight = f"Current trend is {trend}."

    results = {}
    levels = {}

    # Big/Small Prediction
    detected_pattern_bs_info = analyze_complex_sequence(last_results, 'bigsmall', current_volatility=volatility)
    detected_pattern_bs = detected_pattern_bs_info['pattern']
    pattern_strength_bs = detected_pattern_bs_info['strength']
    pattern_influence_bs = 0
    if detected_pattern_bs != 'NONE' and pattern_strength_bs > 0:
        pattern_influence_bs = 5
        level3_bs_insight = f"Sophisticated pattern '{detected_pattern_bs}' (Str: {pattern_strength_bs}) found."
    else:
        level3_bs_insight = 'No dominant pattern observed.'

    pred_bs = ''
    if rsi > 70: pred_bs = 'SMALL' # Overbought, predict reversal
    elif rsi < 30: pred_bs = 'BIG' # Oversold, predict reversal
    else: pred_bs = 'BIG' if numbers[0] >= 5 else 'SMALL' # Neutral RSI, use current number bias

    if trend == 'Up' and pred_bs == 'SMALL': pred_bs = 'BIG'
    if trend == 'Down' and pred_bs == 'BIG': pred_bs = 'SMALL'
    if detected_pattern_bs != 'NONE' and pattern_influence_bs > 0: pred_bs = detected_pattern_bs

    result_bs = pred_bs
    probability_bs = base_confidence + (100 - abs(rsi - 50)) / 5 + pattern_influence_bs
    probability_bs = bayesian_adjust(result_bs, probability_bs, history_data, 'bigsmall')

    if len(last_results['bigsmall']) >= 3:
        recent_actuals = last_results['bigsmall'][:3]
        if all(r == recent_actuals[0] for r in recent_actuals):
            last_actual = recent_actuals[0]
            counter_prediction = 'SMALL' if last_actual == 'BIG' else 'BIG'
            if result_bs == last_actual:
                result_bs = counter_prediction
                probability_bs = max(0, probability_bs - 15)
                level3_bs_insight += f" (Reversed due to 3+ consecutive actual {last_actual} results)"
                print(f"RSI_Trend_Engine: Reversed big/small prediction to {result_bs} due to actual streak of {last_actual}")

    results['bigsmall_result'] = result_bs
    results['bigsmall_probability'] = min(99, max(0, round(probability_bs)))
    levels['bigsmall'] = {'level1': level1_insight, 'level2': level2_insight, 'level3': level3_bs_insight}

    return {
        'bigsmall_result': results['bigsmall_result'], 'bigsmall_probability': results['bigsmall_probability'], 'bigsmall_levels': levels['bigsmall']
    }

# NEW META-STRATEGY: LossStreakStrategy
def LossStreakStrategy(num_recent_losses):
    """
    A meta-strategy activated when the system experiences a significant loss streak.
    It attempts to reverse the recent trend or defer to the best performing regular engine.
    Returns predictions for big/small.
    """
    print(f"Activating Loss Streak Strategy after {num_recent_losses} losses.")
    numbers = [int(item['number']) % 10 for item in cached_data] if cached_data else [random.randint(0, 9) for _ in range(20)]
    if len(numbers) < 5:
        print("Not enough data for LossStreakStrategy, falling back to Hybrid.")
        return HybridMethod()

    results = {}
    levels = {}

    # Strong reversal logic if the actual results have been consistently the same
    # Apply to bigsmall
    res_type = 'bigsmall'
    relevant_actual_outcomes = last_results[res_type]
    if len(relevant_actual_outcomes) >= num_recent_losses and all(r == relevant_actual_outcomes[0] for r in relevant_actual_outcomes[:num_recent_losses]):
        last_actual_outcome = relevant_actual_outcomes[0]
        counter_pred = 'SMALL' if last_actual_outcome == 'BIG' else 'BIG'

        results[f'{res_type}_result'] = counter_pred
        probability = 75 + (num_recent_losses * 2)
        results[f'{res_type}_probability'] = bayesian_adjust(counter_pred, probability, history_data, res_type)
        levels[res_type] = {
            'level1': f"Loss Streak Reversal: Predicted {counter_pred} due to {num_recent_losses} consecutive {last_actual_outcome} actual results.",
            'level2': "Attempting to break sequence.",
            'level3': "High confidence in counter-pattern."
        }
        print(f"LossStreakStrategy: Predicting {results[f'{res_type}_result']} (Confidence: {results[f'{res_type}_probability']}%) for {res_type} due to {num_recent_losses} consecutive ACTUAL {last_actual_outcome} results.")
    else:
        # If no specific loss streak pattern for this type, defer to the best regular engine for this type
        best_regular_engine = determine_best_engine() # This will pick the overall best engine
        
        # Call the appropriate engine function dynamically and extract specific result type
        engine_output = None
        if best_regular_engine == 'QuantumAI': engine_output = QuantumAIMethod()
        elif best_regular_engine == 'NeuralNet': engine_output = NeuralNetMethod()
        elif best_regular_engine == 'Fibonacci': engine_output = FibonacciMethod()
        elif best_regular_engine == 'Hybrid': engine_output = HybridMethod()
        elif best_regular_engine == 'ATS_Engine': engine_output = ATS_Engine()
        elif best_regular_engine == 'RSI_Trend_Engine': engine_output = RSI_Trend_Engine()
        
        if engine_output:
            results[f'{res_type}_result'] = engine_output[f'{res_type}_result']
            results[f'{res_type}_probability'] = engine_output[f'{res_type}_probability']
            levels[res_type] = engine_output[f'{res_type}_levels']
        else:
            results[f'{res_type}_result'] = "N/A"
            results[f'{res_type}_probability'] = 0
            levels[res_type] = {'level1': 'Fallback', 'level2': '', 'level3': ''}

    return {
        'bigsmall_result': results['bigsmall_result'], 'bigsmall_probability': results['bigsmall_probability'], 'bigsmall_levels': levels['bigsmall']
    }


# Function to determine the best engine based on recent performance (from HTML)
def determine_best_engine():
    """
    Determines the best prediction engine to use based on recent performance and Q-learning.
    """
    global current_engine, exploration_rate
    fetch_optimized_data() # Ensure we have the latest data for analysis
    data_numbers = [int(item['number']) % 10 for item in cached_data] if cached_data else []

    current_trend = get_trend_direction(data_numbers)
    current_volatility = calculate_volatility(data_numbers)
    
    # Determine current market regime for contextual performance tracking
    market_regime = "Neutral"
    if current_volatility > 3.0:
        market_regime = "Volatile"
    elif current_volatility < 1.0:
        market_regime = "Stable"
    
    if market_regime == "Stable" and (current_trend == "Up" or current_trend == "Down"):
        market_regime = "Stable Trending"
    elif market_regime == "Volatile" and current_trend == "Sideways":
        market_regime = "Volatile Ranging"
    elif market_regime == "Volatile" and (current_trend == "Up" or current_trend == "Down"):
        market_regime = "Volatile Trending"


    # Update performance metrics for all engines, including per-regime
    for engine in engines:
        engine_wins_overall = 0
        engine_losses_overall = 0
        current_engine_loss_streak_overall = 0

        # Initialize or update regime-specific performance
        for regime_key in engine_performance[engine]:
            if regime_key != 'overall': # Skip overall as it's calculated separately
                engine_performance[engine][regime_key]['wins'] = 0
                engine_performance[engine][regime_key]['losses'] = 0
                engine_performance[engine][regime_key]['accuracy'] = 0
                engine_performance[engine][regime_key]['loss_streak'] = 0

        # Consider only recent history for engine performance evaluation
        engine_history_recent = [item for item in history_data if item.get('server') == engine and item.get('status') in ['WIN 🎉', 'LOSS ❌']][:20] # Longer history for performance tracking

        for item in engine_history_recent:
            is_win = (item['status'] == 'WIN 🎉') # Check actual status string
            
            engine_wins_overall += 1 if is_win else 0
            engine_losses_overall += 1 if not is_win else 0
            current_engine_loss_streak_overall = 0 if is_win else (current_engine_loss_streak_overall + 1)

            # Update regime-specific performance
            # Need to find the historical numbers for the specific period to calculate volatility and trend
            # This is a simplification; a more robust system would store these features with the history entry
            historical_numbers_for_period = [item['number_result'] for item in history_data if item['period'] == item['period'] and 'number_result' in item] # Use number_result from history_data
            item_volatility = calculate_volatility(historical_numbers_for_period) 
            item_trend = get_trend_direction(historical_numbers_for_period)

            item_regime = "Neutral"
            if item_volatility > 3.0: item_regime = "Volatile"
            elif item_volatility < 1.0: item_regime = "Stable"
            
            if item_regime == "Stable" and (item_trend == "Up" or item_trend == "Down"):
                item_regime = "Stable Trending"
            elif item_regime == "Volatile" and item_trend == "Sideways":
                item_regime = "Volatile Ranging"
            elif item_regime == "Volatile" and (item_trend == "Up" or item_trend == "Down"):
                item_regime = "Volatile Trending"

            # Update relevant regime
            if item_regime in engine_performance[engine]:
                reg_stats = engine_performance[engine][item_regime]
                if is_win:
                    reg_stats['wins'] += 1
                    reg_stats['loss_streak'] = 0
                else:
                    reg_stats['losses'] += 1
                    reg_stats['loss_streak'] += 1
                reg_stats['accuracy'] = (reg_stats['wins'] / (reg_stats['wins'] + reg_stats['losses'])) if (reg_wins + reg_losses) > 0 else 0


        engine_performance[engine]['overall']['wins'] = engine_wins_overall
        engine_performance[engine]['overall']['losses'] = engine_losses_overall
        engine_performance[engine]['overall']['accuracy'] = (engine_wins_overall / (engine_wins_overall + engine_losses_overall)) if (engine_wins_overall + engine_losses_overall) > 0 else 0
        engine_performance[engine]['overall']['loss_streak'] = current_engine_loss_streak_overall

        # Calculate accuracy for each regime
        for regime_key in engine_performance[engine]:
            if regime_key != 'overall':
                reg_wins = engine_performance[engine][regime_key]['wins']
                reg_losses = engine_performance[engine][regime_key]['losses']
                engine_performance[engine][regime_key]['accuracy'] = (reg_wins / (reg_wins + reg_losses)) if (reg_wins + reg_losses) > 0 else 0
    
    sorted_engines = sorted(engines, key=lambda x: engine_performance[x]['overall']['accuracy'], reverse=True)

    next_engine = current_engine # Default to current engine

    volatility_level = 'MEDIUM'
    if current_volatility < 1.5: volatility_level = 'LOW'
    elif current_volatility > 3: volatility_level = 'HIGH'

    engine_status = 'NEUTRAL'
    if current_engine in engine_performance:
        if engine_performance[current_engine]['overall']['accuracy'] > 0.6: engine_status = 'WINNING'
        elif engine_performance[current_engine]['overall']['loss_streak'] >= 2: engine_status = 'LOSING'

    # Adjust exploration rate based on overall system performance
    overall_win_rate = (total_wins / total_bets) if total_bets > 0 else 0
    if overall_win_rate < 0.5:
        exploration_rate = 0.2 # More exploration if overall performance is low
        print("Overall win rate low, increasing exploration.")
    elif overall_win_rate > 0.7:
        exploration_rate = 0.05 # Less exploration if overall performance is high
        print("Overall win rate high, decreasing exploration.")
    else:
        exploration_rate = 0.1 # Default exploration

    # Get action from Q-learning (meta-learning)
    current_q_state = get_q_state_improved(predictions_history[0]['status'] if predictions_history else 'NONE', win_count if win_count > 0 else -loss_streak, 'bigsmall', volatility_level, current_trend, engine_status)
    q_learned_action = get_q_action(current_q_state)

    # Apply Q-learning recommendation or switch based on rules
    if random.random() >= exploration_rate: # Exploit Q-learning
        next_engine = q_learned_action
        print(f"Q-Learning recommended engine: {q_learned_action}")
    else: # Explore or apply heuristic switching
        # Prioritize engines that perform well in the *current market regime*
        best_regime_engine = None
        # Check if regime data exists and has enough samples (e.g., > 3 predictions in this regime)
        if market_regime in engine_performance[sorted_engines[0]] and \
           (engine_performance[sorted_engines[0]][market_regime]['wins'] + engine_performance[sorted_engines[0]][market_regime]['losses']) > 3:
            
            sorted_by_regime = sorted(engines, key=lambda x: engine_performance[x][market_regime]['accuracy'], reverse=True)
            if engine_performance[sorted_by_regime[0]][market_regime]['accuracy'] > 0.6: # Only trust if good accuracy in this regime
                best_regime_engine = sorted_by_regime[0]
        
        if best_regime_engine and best_regime_engine != current_engine:
            next_engine = best_regime_engine
            print(f"Switching to {next_engine} due to strong performance in current '{market_regime}' regime.")
        elif current_engine in engine_performance and engine_performance[current_engine]['overall']['loss_streak'] >= 2:
            # If current engine is on a loss streak, try to switch to a better performing one
            switched = False
            for engine in sorted_engines:
                if engine_performance[engine]['overall']['loss_streak'] < 2: # Prioritize engines not on a loss streak
                    next_engine = engine
                    switched = True
                    print(f"Switching from {current_engine} due to loss streak. New engine: {next_engine}")
                    break
                if not switched: # If all engines have loss streaks, pick the one with best overall accuracy
                    next_engine = sorted_engines[0]
                    print(f"All engines have loss streaks. Picking best overall accuracy: {next_engine}")
        else:
            # Otherwise, consider switching to a significantly more accurate engine
            best_overall_engine = sorted_engines[0]
            if engine_performance[best_overall_engine]['overall']['accuracy'] > engine_performance[current_engine]['overall']['accuracy'] + 0.1 and engine_performance[best_overall_engine]['overall']['loss_streak'] < 2:
                next_engine = best_overall_engine
                print(f"Switching to higher accuracy engine: {next_engine}")
    
    print("Final selected engine:", next_engine, "Current Trend:", current_trend, "Volatility:", f"{current_volatility:.2f}", "Market Regime:", market_regime)
    current_engine = next_engine # Update global current engine
    return next_engine

# Generate prediction result with auto-switching
def determine_best_engine_and_generate_result():
    """
    Coordinates the prediction process, selecting the best engine and generating a new prediction.
    """
    global is_result_processing, current_result_prediction, pending_result, predictions_history, current_engine, loss_streak
    if is_result_processing:
        print("Prediction already in progress. Skipping.")
        return

    is_result_processing = True

    prediction_results = {
        'bigsmall_result': "N/A", 'bigsmall_probability': 0, 'bigsmall_levels': {'level1': 'No data', 'level2': 'No data', 'level3': 'No data'}
    }
    selected_engine_name = 'Initializing'

    try:
        fetched_data = fetch_optimized_data()
        if not fetched_data:
            print('Not enough data for prediction.')
            is_result_processing = False
            return

        if loss_streak >= 3: # Activate Loss Streak Strategy if current system is on a significant loss streak
            prediction_results = LossStreakStrategy(loss_streak)
            selected_engine_name = 'LossStreakStrategy'
            print(f"Loss streak of {loss_streak} detected. Forcing LossStreakStrategy.")
        else:
            selected_engine_name = determine_best_engine() # Let Q-learning and heuristics pick the engine

            # Call the selected engine's prediction method
            if selected_engine_name == 'QuantumAI': prediction_results = QuantumAIMethod()
            elif selected_engine_name == 'NeuralNet': prediction_results = NeuralNetMethod()
            elif selected_engine_name == 'Fibonacci': prediction_results = FibonacciMethod()
            elif selected_engine_name == 'Hybrid': prediction_results = HybridMethod()
            elif selected_engine_name == 'ATS_Engine': prediction_results = ATS_Engine()
            elif selected_engine_name == 'RSI_Trend_Engine': prediction_results = RSI_Trend_Engine()
            else: 
                prediction_results = {
                    'bigsmall_result': "N/A", 'bigsmall_probability': 0, 'bigsmall_levels': {'level1': 'N/A', 'level2': 'N/A', 'level3': 'N/A'}
                }

        if prediction_results['bigsmall_result'] != "N/A":
            new_prediction_entry = {
                "period": last_period_number,
                "bigsmall_prediction": prediction_results['bigsmall_result'],
                "bigsmall_confidence": prediction_results['bigsmall_probability'],
                "optimal_bet": suggest_optimal_bet(prediction_results['bigsmall_probability'], loss_streak), # Use bigsmall confidence for bet
                "bigsmall_result": "...waiting",
                "status": "Pending",
                "server": selected_engine_name, # Record which engine made the prediction
                "bigsmall_levels": prediction_results['bigsmall_levels'],
                'nn_input_features': prediction_results.get('nn_input_features', {}), # Store NN input features
                'nn_prediction_strength': prediction_results.get('nn_prediction_strength', 0.0) # Store NN raw strength
            }
            predictions_history.insert(0, new_prediction_entry)
            if len(predictions_history) > 50: # Keep history manageable
                predictions_history.pop()
            pending_result = predictions_history[0] # Set the latest prediction as pending
            print(f"Prediction generated for Period {last_period_number}: Big/Small: {prediction_results['bigsmall_result']} ({prediction_results['bigsmall_probability']}%) by {selected_engine_name}")
    except Exception as e:
        print(f"Error in determineBestEngineAndGenerateResult: {e}")
    finally:
        is_result_processing = False

# Function to update NeuralNet weights based on a batch of errors
def _update_neural_net_weights(batch_inputs, batch_errors, nn_learning_rate):
    """
    Applies a simulated batch update to NeuralNet weights using momentum and weight decay.
    """
    global neural_net_internal_weights, neural_net_feature_importance_scores, \
           nn_activation_alpha, nn_activation_beta

    if not batch_inputs:
        return

    # Average the error signals across the batch
    avg_error_signal = sum(batch_errors) / len(batch_errors)

    # Average the input features across the batch (for a more stable gradient approximation)
    avg_nn_input_features = {}
    if batch_inputs: # Ensure batch_inputs is not empty before iterating keys
        for feature_name in batch_inputs[0].keys():
            avg_nn_input_features[feature_name] = sum(inp.get(feature_name, 0.0) for inp in batch_inputs) / len(batch_inputs)

    # Apply updates to each weight
    for key, weight_data in neural_net_internal_weights.items():
        input_val = 0.0 # Default input value for this weight
        
        # Heuristic mapping from weight key to its corresponding average input feature value
        if key == 'bias': input_val = 1.0
        elif key == 'w_trend': input_val = avg_nn_input_features.get('f1_trend', 0.0)
        elif key == 'w_parity': input_val = avg_nn_input_features.get('f2_parity', 0.0)
        elif key == 'w_volatility': input_val = avg_nn_input_features.get('f3_volatility_norm', 0.0)
        elif key == 'w_win_streak': input_val = avg_nn_input_features.get('f4_win_streak_norm', 0.0)
        elif key == 'w_loss_streak': input_val = avg_nn_input_features.get('f5_loss_streak_norm', 0.0)
        elif key == 'w_time_remaining': input_val = avg_nn_input_features.get('f6_time_remaining_norm', 0.0)
        elif key == 'w_pattern': input_val = avg_nn_input_features.get('f7_pattern_strength_bs_norm', 0.0) # Using BS pattern as general pattern
        elif key == 'w_prev_pred_strength': input_val = avg_nn_input_features.get('f8_prev_pred_strength', 0.0)
        elif key == 'w_prev_actual_outcome': input_val = avg_nn_input_features.get('f9_prev_actual_outcome_val', 0.0)
        elif key == 'w_recurrent_avg_input_strength': input_val = avg_nn_input_features.get('f10_recurrent_avg_input_strength', 0.0)
        elif key == 'w_recurrent_last_trend_val': input_val = avg_nn_input_features.get('f11_recurrent_last_trend_val', 0.0)
        elif key == 'w_recurrent_error_memory': input_val = avg_nn_input_features.get('f12_recurrent_error_memory', 0.0)
        elif key == 'w_recurrent_volatility_fingerprint': input_val = avg_nn_input_features.get('f13_recurrent_volatility_fingerprint', 0.0)
        elif key == 'w_market_sentiment': 
            sentiment_score = (avg_nn_input_features.get('rsi', 50) - 50) / 50.0 + avg_nn_input_features.get('ma_bias', 0.0) / 5.0
            input_val = sentiment_score
        elif key == 'w_trend_confirmation': 
            trend_conf = avg_nn_input_features.get('f1_trend', 0.0) * avg_nn_input_features.get('f3_volatility_norm', 0.0)
            input_val = trend_conf
        # New cross-feature interaction weights
        elif key == 'w_trend_volatility_interaction': input_val = avg_nn_input_features.get('f1_trend', 0.0) * avg_nn_input_features.get('f3_volatility_norm', 0.0)
        elif key == 'w_streak_time_interaction': input_val = avg_nn_input_features.get('f4_win_streak_norm', 0.0) * avg_nn_input_features.get('f6_time_remaining_norm', 0.0)
        elif key == 'w_parity_pattern_interaction': input_val = avg_nn_input_features.get('f2_parity', 0.0) * avg_nn_input_features.get('f7_pattern_strength_bs_norm', 0.0)
        # For hidden layer weights, this is a simplification; in real NN, it's more complex
        # We'll use a proxy for hidden layer activations (e.g., average prediction strength)
        elif key.startswith('w_hidden') or key.startswith('w_output_from_hidden'):
            input_val = sum(inp.get('nn_prediction_strength', 0.0) for inp in batch_inputs) / len(batch_inputs) # Average raw strength as proxy

        # Heuristic gradient approximation for this weight
        # Scale error signal by confidence of the *incorrect* prediction
        scaled_error_signal = avg_error_signal * (1 - abs(avg_error_signal)) # Stronger signal for clearer errors
        gradient_approximation = scaled_error_signal * input_val
        
        # Adaptive Momentum: scale momentum factor by NN's accuracy (more momentum if doing well)
        current_momentum_factor = NN_MOMENTUM_FACTOR
        nn_accuracy_for_momentum = engine_performance['NeuralNet']['overall']['accuracy']
        if nn_accuracy_for_momentum > 0.6:
            current_momentum_factor = min(0.95, NN_MOMENTUM_FACTOR + (nn_accuracy_for_momentum - 0.6) * 0.1)
        elif nn_accuracy_for_momentum < 0.4:
            current_momentum_factor = max(0.5, NN_MOMENTUM_FACTOR - (0.4 - nn_accuracy_for_momentum) * 0.1)

        # Apply momentum: new_adjustment = learning_rate * gradient + momentum_factor * last_adjustment
        new_adjustment = nn_learning_rate * gradient_approximation + current_momentum_factor * weight_data['last_adj']
        
        # Simulated Gradient Clipping
        clip_threshold = 0.1 # Max allowed adjustment in one step
        new_adjustment = max(-clip_threshold, min(clip_threshold, new_adjustment))

        weight_data['value'] += new_adjustment
        weight_data['last_adj'] = new_adjustment # Store for next momentum calculation
        
        # Apply weight decay (L2 regularization)
        # Dynamically adjust weight decay based on NN's accuracy
        current_weight_decay = NN_WEIGHT_DECAY
        if nn_accuracy_for_momentum > 0.8: # If very accurate, slightly increase decay to prevent overfitting
            current_weight_decay *= 1.5
        elif nn_accuracy_for_momentum < 0.5: # If struggling, slightly decrease decay to allow more aggressive learning
            current_weight_decay *= 0.5

        weight_data['value'] -= current_weight_decay * weight_data['value']

        # Clamp weights to prevent explosion
        weight_data['value'] = max(-1.0, min(1.0, weight_data['value']))

        # Update feature importance scores (heuristic: based on absolute change in weight)
        if key.startswith('w_') and not key.startswith(('w_hidden', 'w_output_from_hidden', 'bias')):
            feature_name_map = {
                'w_trend': 'f1_trend', 'w_parity': 'f2_parity', 'w_volatility': 'f3_volatility_norm',
                'w_win_streak': 'f4_win_streak_norm', 'w_loss_streak': 'f5_loss_streak_norm',
                'w_time_remaining': 'f6_time_remaining_norm', 'w_pattern': 'f7_pattern_strength_norm',
                'w_prev_pred_strength': 'f8_prev_pred_strength', 'w_prev_actual_outcome': 'f9_prev_actual_outcome_val',
                'w_recurrent_avg_input_strength': 'f10_recurrent_avg_input_strength',
                'w_recurrent_last_trend_val': 'f11_recurrent_last_trend_val',
                'w_recurrent_error_memory': 'f12_recurrent_error_memory',
                'w_recurrent_volatility_fingerprint': 'f13_recurrent_volatility_fingerprint',
                'w_market_sentiment': 'market_sentiment', 'w_trend_confirmation': 'trend_confirmation',
                'w_trend_volatility_interaction': 'trend_volatility_interaction',
                'w_streak_time_interaction': 'streak_time_interaction',
                'w_parity_pattern_interaction': 'parity_pattern_interaction'
            }
            mapped_feature_name = feature_name_map.get(key)
            if mapped_feature_name:
                # Increase importance if weight changed significantly
                neural_net_feature_importance_scores[mapped_feature_name] += abs(new_adjustment) * 0.01
                # Decay importance over time
                for f_key in neural_net_feature_importance_scores:
                    neural_net_feature_importance_scores[f_key] *= 0.99 # Slight decay
                neural_net_feature_importance_scores[mapped_feature_name] = min(1.0, neural_net_feature_importance_scores[mapped_feature_name]) # Clamp

    # Update activation function parameters (alpha and beta)
    # Heuristic: adjust alpha and beta to make activation more sensitive/less sensitive based on error
    nn_activation_alpha['value'] += avg_error_signal * nn_learning_rate * 0.01 # Adjust range
    nn_activation_alpha['value'] = max(0.5, min(2.0, nn_activation_alpha['value'])) # Clamp alpha
    
    nn_activation_beta['value'] += avg_error_signal * nn_learning_rate * 0.02 # Adjust steepness
    nn_activation_beta['value'] = max(0.5, min(2.0, nn_activation_beta['value'])) # Clamp beta

    print(f"NeuralNet: Weights updated for batch. Avg Error: {avg_error_signal:.2f}, Learning Rate: {nn_learning_rate:.3f}, Momentum: {current_momentum_factor:.2f}, Alpha: {nn_activation_alpha['value']:.2f}, Beta: {nn_activation_beta['value']:.2f}")

# Check pending result against actual game result
def check_pending_result(period):
    """
    Checks if a pending prediction has been resolved by an actual game result.
    Updates statistics, Q-table, and Telegram messages.
    """
    global pending_result, win_count, loss_streak, total_bets, total_wins, total_losses, history_data, \
           neural_net_previous_prediction_strength, neural_net_previous_actual_outcome, \
           neural_net_internal_weights, neural_net_recurrent_state, \
           nn_batch_inputs, nn_batch_errors

    # Find the pending prediction in history
    history_entry_index = next((i for i, h in enumerate(predictions_history) if h['period'] == period and h['status'] == 'Pending'), -1)
    if history_entry_index == -1:
        return

    current_pending_result = predictions_history[history_entry_index]
    api_result = fetch_latest_actual_result(period)

    if api_result and api_result['period'] == period:
        # Check big/small prediction
        is_win_bs = (current_pending_result['bigsmall_prediction'] == api_result['bigsmall_result'])
        
        # Overall win/loss for the period is based on big/small prediction
        is_overall_win = is_win_bs
        
        current_pending_result['status'] = "WIN 🎉" if is_overall_win else "LOSS ❌"
        current_pending_result['bigsmall_result'] = api_result['bigsmall_result']
        current_pending_result['actualNumber'] = api_result['number_result'] # Keep for volatility/trend calculation

        print(f"Period {period} resolved.")
        print(f"  Predicted BS: {current_pending_result['bigsmall_prediction']}, Actual BS: {current_pending_result['bigsmall_result']}. Status: {'WIN 🎉' if is_win_bs else 'LOSS ❌'}")
        print(f"  Overall Period Status: {current_pending_result['status']}")

        # Update last_results for pattern analysis for bigsmall
        last_results['bigsmall'].insert(0, current_pending_result['bigsmall_result'])
        if len(last_results['bigsmall']) > max_history_items_for_analysis:
            last_results['bigsmall'].pop()
        
        # Add resolved prediction to history_data for Bayesian adjustment and Loss Streak Strategy
        history_data.insert(0, {
            "period": current_pending_result['period'],
            "bigsmall_prediction": current_pending_result['bigsmall_prediction'],
            "bigsmall_result": current_pending_result['bigsmall_result'],
            "status": current_pending_result['status'], # Overall status
            "server": current_pending_result['server'],
            "bigsmall_confidence": current_pending_result['bigsmall_confidence'],
            "number_result": current_pending_result['actualNumber'] # Store actual number for historical volatility/trend
        })
        if len(history_data) > max_history_items_for_analysis:
            history_data.pop()

        # Update overall win/loss streaks and counts
        if is_overall_win:
            win_count += 1
            loss_streak = 0
            total_wins += 1
            send_sticker(WIN_STICKER)
        else:
            loss_streak += 1
            win_count = 0
            total_losses += 1
            send_sticker(LOSS_STICKER)
        
        total_bets += 1

        # Prepare states for Q-learning update
        data_numbers = [item['number_result'] for item in history_data if 'number_result' in item] # Use number_result from history_data
        current_volatility = calculate_volatility(data_numbers)
        volatility_level = 'MEDIUM'
        if current_volatility < 1.5: volatility_level = 'LOW'
        elif current_volatility > 3: volatility_level = 'HIGH'

        current_trend = get_trend_direction(data_numbers)

        engine_status = 'NEUTRAL'
        if current_engine in engine_performance:
            if engine_performance[current_engine]['overall']['accuracy'] > 0.6: engine_status = 'WINNING'
            elif engine_performance[current_engine]['overall']['loss_streak'] >= 2: engine_status = 'LOSING'
        elif current_pending_result['server'] == 'LossStreakStrategy':
            engine_status = 'ADAPTIVE' # Special status for meta-strategy
        
        reward = 1 if is_overall_win else -1

        # Get previous state for Q-learning (state *before* this prediction was made)
        prev_state_for_q = get_q_state_improved(
            predictions_history[history_entry_index + 1]['status'] if len(predictions_history) > (history_entry_index + 1) else 'NONE',
            win_count - (1 if is_overall_win else 0) if win_count > 0 else -(loss_streak - (1 if not is_overall_win else 0)), # Streak *before* this result
            'bigsmall', # Q-learning state for bigsmall for now
            volatility_level,
            current_trend,
            engine_status
        )

        # Get current (next) state for Q-learning (state *after* this prediction's outcome)
        next_state_for_q = get_q_state_improved(
            current_pending_result['status'],
            win_count if win_count > 0 else -loss_streak, # Current streak
            'bigsmall', # Q-learning state for bigsmall for now
            volatility_level,
            current_trend,
            engine_status
        )

        # Update Q-table if the prediction was made by a "learning" engine (not a meta-strategy)
        if current_pending_result['server'] in engines: 
            update_q_table(prev_state_for_q, current_pending_result['server'], reward, next_state_for_q)
        else:
            print(f"Q-Learning: Skipping Q-table update for non-learning engine/strategy: {current_pending_result['server']}")

        # Update individual engine performance metrics
        if current_pending_result['server'] not in engine_performance:
            # Initialize all sub-dictionaries if engine is new
            engine_performance[current_pending_result['server']] = {
                'overall': {'wins': 0, 'losses': 0, 'accuracy': 0, 'loss_streak': 0},
                'volatile': {'wins': 0, 'losses': 0, 'accuracy': 0, 'loss_streak': 0},
                'stable': {'wins': 0, 'losses': 0, 'accuracy': 0, 'loss_streak': 0},
                'trending': {'wins': 0, 'losses': 0, 'accuracy': 0, 'loss_streak': 0},
                'ranging': {'wins': 0, 'losses': 0, 'accuracy': 0, 'loss_streak': 0},
            }

        # Update overall performance
        if is_overall_win:
            engine_performance[current_pending_result['server']]['overall']['wins'] += 1
            engine_performance[current_pending_result['server']]['overall']['loss_streak'] = 0
        else:
            engine_performance[current_pending_result['server']]['overall']['losses'] += 1
            engine_performance[current_pending_result['server']]['overall']['loss_streak'] += 1
        
        engine_performance[current_pending_result['server']]['overall']['accuracy'] = \
            (engine_performance[current_pending_result['server']]['overall']['wins'] / (engine_performance[current_pending_result['server']]['overall']['wins'] + engine_performance[current_pending_result['server']]['overall']['losses'])) \
            if (engine_performance[current_pending_result['server']]['overall']['wins'] + engine_performance[current_pending_result['server']]['overall']['losses']) > 0 else 0
        
        # Update regime-specific performance
        current_numbers_for_regime = [item['number_result'] for item in history_data if 'number_result' in item] # Use number_result from history_data
        current_volatility_for_regime = calculate_volatility(current_numbers_for_regime)
        current_trend_for_regime = get_trend_direction(current_numbers_for_regime)

        current_market_regime_key = "Neutral"
        if current_volatility_for_regime > 3.0: current_market_regime_key = "Volatile"
        elif current_volatility_for_regime < 1.0: current_market_regime_key = "Stable"
        
        if current_market_regime_key == "Stable" and (current_trend_for_regime == "Up" or current_trend_for_regime == "Down"):
            current_market_regime_key = "Stable Trending"
        elif current_market_regime_key == "Volatile" and current_trend_for_regime == "Sideways":
            current_market_regime_key = "Volatile Ranging"
        elif current_market_regime_key == "Volatile" and (current_trend_for_regime == "Up" or current_trend_for_regime == "Down"):
            current_market_regime_key = "Volatile Trending"

        # Update the relevant regime's stats
        if current_market_regime_key in engine_performance[current_pending_result['server']]:
            reg_stats = engine_performance[current_pending_result['server']][current_market_regime_key]
            if is_overall_win:
                reg_stats['wins'] += 1
                reg_stats['loss_streak'] = 0
            else:
                reg_stats['losses'] += 1
                reg_stats['loss_streak'] += 1
            reg_stats['accuracy'] = (reg_stats['wins'] / (reg_stats['wins'] + reg_stats['losses'])) if (reg_stats['wins'] + reg_stats['losses']) > 0 else 0

        print(f"Updated performance for {current_pending_result['server']}: Overall Accuracy: {engine_performance[current_pending_result['server']]['overall']['accuracy']:.2f}, Loss Streak: {engine_performance[current_pending_result['server']]['overall']['loss_streak']}")
        print(f"  Regime '{current_market_regime_key}' Accuracy: {engine_performance[current_pending_result['server']][current_market_regime_key]['accuracy']:.2f}")


        identify_unexpected_result(current_pending_result['bigsmall_prediction'], current_pending_result['bigsmall_result'], current_pending_result['bigsmall_confidence'])
        pending_result = None # Clear pending result after processing

        # --- Update NeuralNet's internal memory and weights after a prediction is resolved ---
        if current_pending_result['server'] == 'NeuralNet':
            # Store previous prediction strength and actual outcome for next cycle's memory
            neural_net_previous_prediction_strength = current_pending_result['nn_prediction_strength'] # Use raw strength
            neural_net_previous_actual_outcome = current_pending_result['status'] # 'WIN' or 'LOSS'
            print(f"NeuralNet: Updated internal memory. Prev Strength: {neural_net_previous_prediction_strength:.2f}, Prev Outcome: {neural_net_previous_actual_outcome}")

            # Simulated Backpropagation for NeuralNet's internal weights
            # This is a heuristic adjustment based on error, mimicking gradient descent
            if current_pending_result['bigsmall_confidence'] > 70: # Only adjust weights significantly if high confidence
                # Calculate a target output for the prediction strength (e.g., +1 for win, -1 for loss)
                target_output = 1.0 if is_overall_win else -1.0
                # Calculate a simple error based on the NN's raw prediction strength
                error_signal = target_output - current_pending_result['nn_prediction_strength'] 
                
                # Accumulate inputs and errors for batch update
                nn_batch_inputs.append(current_pending_result['nn_input_features'])
                nn_batch_errors.append(error_signal)

                # Trigger batch update if batch size is reached
                # Dynamic NN_BATCH_SIZE: smaller in high volatility, larger in low volatility
                current_volatility_for_batch = calculate_volatility([item['number_result'] for item in history_data if 'number_result' in item]) # Use number_result from history_data
                dynamic_nn_batch_size = NN_BATCH_SIZE
                if current_volatility_for_batch > 3.0: # High volatility
                    dynamic_nn_batch_size = max(1, NN_BATCH_SIZE // 2) # Smaller batch for faster adaptation
                elif current_volatility_for_batch < 1.0: # Low volatility
                    dynamic_nn_batch_size = NN_BATCH_SIZE * 2 # Larger batch for more stable learning

                if len(nn_batch_inputs) >= dynamic_nn_batch_size:
                    nn_accuracy_for_learning = engine_performance['NeuralNet']['overall']['accuracy']
                    nn_learning_rate_for_update = 0.05 # Base
                    if nn_accuracy_for_learning < 0.5:
                        nn_learning_rate_for_update += (0.5 - nn_accuracy_for_learning) * 0.1
                    elif nn_accuracy_for_learning > 0.7:
                        nn_learning_rate_for_update -= (nn_accuracy_for_learning - 0.7) * 0.05
                    nn_learning_rate_for_update = max(0.01, min(0.2, nn_learning_rate_for_update))

                    _update_neural_net_weights(nn_batch_inputs, nn_batch_errors, nn_learning_rate_for_update)
                    nn_batch_inputs.clear() # Clear batch after update
                    nn_batch_errors.clear()

                # Update recurrent error memory
                neural_net_recurrent_state['error_memory'] = (neural_net_recurrent_state['error_memory'] * 0.5) + (error_signal * 0.5) # Decay and add new error
                neural_net_recurrent_state['long_term_error_avg'] = (neural_net_recurrent_state['long_term_error_avg'] * 0.9) + (error_signal * 0.1) # Slower decay for long term
            else:
                # Decay error memory even on wins or low-confidence losses
                neural_net_recurrent_state['error_memory'] *= 0.8 
                neural_net_recurrent_state['long_term_error_avg'] *= 0.95 # Slower decay for long term

            # Update recurrent state based on the actual outcome
            current_input_strength = (current_pending_result['bigsmall_confidence'] - 50) / 50.0
            neural_net_recurrent_state['avg_input_strength'] = (neural_net_recurrent_state['avg_input_strength'] * 0.8) + (current_input_strength * 0.2) # Exponential moving average
            
            # Update last trend value in recurrent state
            numbers_for_trend = [item['number_result'] for item in history_data if 'number_result' in item] # Use number_result from history_data
            current_trend_val = 1 if get_trend_direction(numbers_for_trend) == 'Up' else (-1 if get_trend_direction(numbers_for_trend) == 'Down' else 0)
            neural_net_recurrent_state['last_trend_val'] = current_trend_val

            # Update volatility fingerprint and trend in recurrent state
            prev_volatility = neural_net_recurrent_state['volatility_fingerprint']
            current_volatility_for_fp = calculate_volatility(numbers_for_trend)
            neural_net_recurrent_state['volatility_fingerprint'] = (neural_net_recurrent_state['volatility_fingerprint'] * 0.7) + (current_volatility_for_fp * 0.3)
            neural_net_recurrent_state['volatility_trend'] = (neural_net_recurrent_state['volatility_trend'] * 0.8) + ((current_volatility_for_fp - prev_volatility) * 0.2)


            print(f"NeuralNet Recurrent State Updated: Avg Input Strength: {neural_net_recurrent_state['avg_input_strength']:.2f}, Last Trend: {neural_net_recurrent_state['last_trend_val']}, Error Memory: {neural_net_recurrent_state['error_memory']:.2f}, Volatility FP: {neural_net_recurrent_state['volatility_fingerprint']:.2f}, Long Term Error: {neural_net_recurrent_state['long_term_error_avg']:.2f}, Volatility Trend: {neural_net_recurrent_state['volatility_trend']:.2f}")


        # --- Telegram Edit for Result ---
        msg_id = telegram_msg_ids.get(current_pending_result['period'])
        if use_bot and msg_id:
            total = total_wins + total_losses
            acc = (total_wins / total) * 100 if total else 0
            
            status_emoji = '🎉' if is_overall_win else '❌'
            
            # Get current date for the message
            now = datetime.now(timezone.utc)
            current_date_str = now.strftime("%d-%m-%Y")

            edited_msg_text = (
                f"╔═◈═◈═◈═◈═◈═╗Date : {current_date_str}\n"
                f"Accurate. Fast. AI-Powered Wingo : 1MINUTEPeriod No : {current_pending_result['period']}\n"
                f"╚═◈═◈═◈═◈═◈═╝ RESULT INFO \n"
                f"Big/Small: {current_pending_result['bigsmall_result']}\n"
                f"──────✦✧✦──────\n"
                f"STATUS -> {current_pending_result['status']}\n"
                f"STATS -> WINS: {total_wins} LOSS: {total_losses} ACCURACY: {acc:.2f}%\n"
            )
            edit_telegram_message(msg_id, edited_msg_text)

    else:
        print(f"Actual result for period {period} not yet available from API. Keeping pending.")


# ==== Terminal Colors ====
def color(text, code): return f"\033[{code}m{text}\033[0m"

# ==== Telegram Bot Functions (unchanged, for completeness) ====
def send_telegram_message(text):
    if use_bot:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            res = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}) # Added parse_mode
            return res.json().get("result", {}).get("message_id")
        except Exception as e: 
            print(f"Telegram send error: {e}")
            return None

def edit_telegram_message(message_id, text):
    if use_bot and message_id:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/editMessageText"
            requests.post(url, data={
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": "HTML" # Added parse_mode
            })
        except Exception as e: 
            print(f"Telegram edit error: {e}")
            pass

def send_sticker(sticker_id):
    if use_bot:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendSticker"
            requests.post(url, data={"chat_id": chat_id, "sticker": sticker_id})
        except Exception as e:
            print(f"Telegram sticker error: {e}")
            pass

# ==== Display Terminal ====
def display():
    """
    Clears the console and displays the current prediction status and history.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print(color("┌────────────────────────────────────────────┐", "96"))
    # Updated name to AWD PYTHON AI
    print(color("│             AWD PYTHON AI                  │", "96")) 
    print(color("│  🧠✨ Highly Advanced Predictions Engine ✨🤖   │", "96")) 
    print(color("└────────────────────────────────────────────┘\n", "96"))

    current_period_val = get_period()

    # --- TOP SECTION: Current Prediction ---
    if predictions_history:
        current_pred_info = predictions_history[0]
        optimal_bet_display = current_pred_info.get("optimal_bet", "-- units")

        print(f"  CURRENT PERIOD     : {color(current_period_val, '36')}")
        print(f"  PREDICTION (BS)    → {color(current_pred_info['bigsmall_prediction'], '93')} ({current_pred_info['bigsmall_confidence']}%)")
        print(f"  OPTIMAL BET        : {color(optimal_bet_display, '93')}\n")
    else:
        print(f"  CURRENT PERIOD     : {color(current_period_val, '36')}")
        print(f"  PREDICTION (BS)    → {color('--', '93')}")
        print(f"  OPTIMAL BET        : {color('-- units', '93')}\n")

    # --- ANALYSIS DASHBOARD ---
    acc_dashboard = (total_wins / total_bets) * 100 if total_bets else 0

    print(color("┌────────────────ANALYSIS DASHBOARD───────────┐", "90"))
    print(f"│ WINS: {color(str(total_wins), '92'):<5} | LOSSES: {color(str(total_losses), '91'):<5} | SKIPS: {color(str(sum(1 for p in predictions_history if p['status'] == 'SKIP')), '95'):<5} | ACCURACY: {color(f'{acc_dashboard:.2f}%', '94'):<8} │")
    print(f"│ CURRENT ENGINE: {color(current_engine, '96'):<30}│")
    print(color("└────────────────────────────────────────────┘\n", "90"))

    # --- PREDICTION HISTORY (updated to match Telegram format) ---
    print(color("┌────────────────PREDICTION HISTORY───────────┐", "90"))
    for p in predictions_history[:5]: # Display last 5 predictions
        status_emoji = ''
        if p['status'] == "WIN �":
            status_emoji = ' 🎉'
        elif p['status'] == "LOSS ❌":
            status_emoji = ' ❌'
        
        actual_display_text_bs = f"ACTUAL BS → {p['bigsmall_result']}" if p['status'] != "Pending" else ""

        print(f"│🎯{p['period']}")
        print(f"│PREDICTION BS → {color(p['bigsmall_prediction'], '93')}")

        if actual_display_text_bs:
            print(f"│{actual_display_text_bs}")
        
        if p['status'] != "Pending":
            current_wins_for_display = total_wins
            current_losses_for_display = total_losses
            current_total_decisions_for_display = current_wins_for_display + current_losses_for_display
            current_accuracy_for_display = (current_wins_for_display / current_total_decisions_for_display) * 100 if current_total_decisions_for_display else 0
            print(f"│STATUS: {p['status']}")
            print(f"│STATS → WINS: {current_wins_for_display} LOSS: {current_losses_for_display} ACCURACY: {current_accuracy_for_display:.2f}%")
        
        print(color("└────────────────────────────────────────────┘", "90"))
        print("\n", end="") # Add a newline to separate boxes visually

    # Added "Powered by" line at the bottom
    print(color("✨ Powered by @WEBDEVSRS ✨", "96"))


# Function to suggest optimal bet based on confidence and streak (from HTML)
def suggest_optimal_bet(confidence, current_loss_streak):
    """
    Suggests an optimal bet size based on prediction confidence and current loss streak.
    Implements a simple risk management strategy.
    """
    bet_units = 1

    # Get overall system accuracy and current market volatility
    overall_system_accuracy = (total_wins / total_bets) if total_bets > 0 else 0
    current_market_volatility = calculate_volatility([int(item['number']) % 10 for item in cached_data]) if cached_data else [] # Ensure numbers are available
    
    # Determine market regime based on volatility and trend
    market_regime = "Neutral"
    if current_market_volatility: # Check if list is not empty before accessing
        if current_market_volatility > 3.0:
            market_regime = "Volatile"
        elif current_market_volatility < 1.0:
            market_regime = "Stable"
    
    current_numbers_for_trend = [int(item['number']) % 10 for item in cached_data] if cached_data else []
    trend_direction = get_trend_direction(current_numbers_for_trend)
    if market_regime == "Stable" and (trend_direction == "Up" or trend_direction == "Down"):
        market_regime = "Stable Trending"
    elif market_regime == "Volatile" and trend_direction == "Sideways":
        market_regime = "Volatile Ranging"
    elif market_regime == "Volatile" and (trend_direction == "Up" or trend_direction == "Down"):
        market_regime = "Volatile Trending"
    
    # Adjust base bet units based on confidence
    if confidence > 90:
        bet_units = 3
    elif confidence > 80:
        bet_units = 2
    elif confidence < 70: # Reduce bet if confidence is low
        bet_units = 0.5

    # Further adjust based on personal loss streak
    if current_loss_streak >= 1:
        bet_units *= 0.5
    if current_loss_streak >= 2:
        bet_units *= 0.5
    if current_loss_streak >= 3:
        bet_units = 0 # Stop betting after 3 consecutive losses

    # Adjust based on overall system accuracy: more aggressive if AI is performing well
    if overall_system_accuracy > 0.65:
        bet_units *= 1.2
    elif overall_system_accuracy < 0.45:
        bet_units *= 0.7

    # Adjust based on market regime: fine-tune risk for different market types
    if market_regime == "Volatile":
        bet_units *= 0.6 # Reduce bets significantly in high volatility
    elif market_regime == "Stable Trending":
        bet_units *= 1.1 # Increase bets slightly in stable trends
    elif market_regime == "Volatile Ranging":
        bet_units *= 0.7 # Cautious in volatile but non-trending markets

    # Ensure a minimum bet if not stopping
    if bet_units > 0 and bet_units < 0.5: bet_units = 0.5

    return 'STOP' if bet_units <= 0.1 else f"{bet_units:.1f} units"


# ==== Main Loop ====
def main():
    """
    Main loop of the prediction engine.
    Continuously checks for new periods, generates predictions, and resolves pending results.
    """
    global last_period_number, pending_result, last_api_time, predictions_history

    if use_bot:
        send_sticker(STARTUP_STICKER)

    try:
        fetch_optimized_data() # Initial data fetch on startup
    except Exception as e:
        print(f"Initial historical data fetch failed: {e}")

    while True:
        current_period_check = get_period()

        # Check if a new period has started
        if last_period_number != current_period_check:
            print(f"New Period Detected: {current_period_check}")
            # If there was a pending result from the previous period, check it now
            if pending_result and last_period_number:
                try:
                    check_pending_result(last_period_number)
                except Exception as e:
                    print(f"Error checking pending result for {last_period_number}: {e}")
            
            last_period_number = current_period_check # Update to the new period
            determine_best_engine_and_generate_result() # Generate a new prediction for the new period
            
            # Send initial prediction message via Telegram if bot is enabled
            if use_bot and predictions_history:
                latest_pred = predictions_history[0]
                now = datetime.now(timezone.utc)
                current_date_str = now.strftime("%d-%m-%Y")
                msg = (
                    f"╔═◈═◈═◈═◈═◈═╗Date : {current_date_str}\n"
                    f"Accurate. Fast. AI-Powered Wingo : 1MINUTEPeriod No : {latest_pred['period']}\n"
                    f"╚═◈═◈═◈═◈═◈═╝ UPCOMING RESULT \n" # Changed to UPCOMING RESULT
                    f"Big/Small: {latest_pred['bigsmall_prediction']} ({latest_pred['bigsmall_confidence']}%)\n"
                    f"──────✦✧✦──────"
                )
                msg_id = send_telegram_message(msg)
                telegram_msg_ids[latest_pred['period']] = msg_id # Store message ID for later edits

        # Continuously check pending result in case API takes time to update
        if pending_result:
            try:
                check_pending_result(pending_result['period'])
            except Exception as e:
                print(f"Error checking pending result for {pending_result['period']}: {e}")

        display() # Update the terminal display
        time.sleep(1) # Wait for 1 second before next loop iteration

if __name__ == "__main__":
    main()