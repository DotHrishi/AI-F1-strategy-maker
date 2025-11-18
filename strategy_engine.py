import numpy as np
import joblib
import requests
import os
from typing import List, Tuple, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load precomputed
deg_models = joblib.load('models/deg_models.pkl')
pit_time = joblib.load('models/pit_time.pkl')
track_laps_dict = joblib.load('models/track_laps.pkl')

# Track coords for weather (lat, lon)
TRACK_COORDS = {
    'Bahrain': (26.0325, 50.5106), 'Saudi Arabia': (26.8528, 43.9719), 'Australia': (-37.8497, 144.8430),
    'Japan': (34.8431, 139.3275), 'China': (31.2862, 121.1825), 'Miami': (25.9581, -80.2389),
    'Imola': (44.3439, 11.7131), 'Monaco': (43.7419, 7.4251), 'Canada': (45.5007, -73.5224),
    'Spain': (41.5700, 2.2611), 'Austria': (47.2194, 14.7964), 'Silverstone': (52.0754, -0.8480),
    'Hungary': (47.5786, 19.2523), 'Belgium': (50.4381, 5.9717), 'Netherlands': (52.3914, 4.5411),
    'Monza': (45.6156, 9.2852), 'Azerbaijan': (40.3725, 49.8533), 'Singapore': (1.2914, 103.8642),
    'USA': (30.1328, -97.6413), 'Mexico': (19.4042, -99.0907), 'Brazil': (-23.7006, -46.6944),
    'Las Vegas': (36.1169, -115.1833), 'Qatar': (25.1497, 51.5111), 'Abu Dhabi': (24.4672, 54.6031)
}

OW_API_KEY = os.getenv('OPENWEATHER_API_KEY')  # Use the correct env var name

def get_weather(track: str) -> Dict:
    """Fetch real-time weather via OpenWeatherMap."""
    if track not in TRACK_COORDS:
        raise ValueError("Unknown track")
    lat, lon = TRACK_COORDS[track]
    api_key = OW_API_KEY  # Use the loaded API key
    if not api_key or api_key == 'demo':
        # Fallback for no key: Neutral dry weather
        print(f"No OpenWeather API key found, using default weather for {track}")
        return {'track_temp': 30.0, 'rain_prob': 0.1}
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        resp = requests.get(url, timeout=10).json()
        if resp.get('cod') != 200:
            print(f"API Error: {resp.get('message', 'Unknown')}")
            return {'track_temp': 30.0, 'rain_prob': 0.1}  # Fallback
        temp = resp['main']['temp']
        track_temp = temp + 10  # Approx track temp (track is usually warmer than air)
        rain = resp.get('rain', {}).get('1h', 0)
        rain_prob = min(1.0, rain / 5)  # Normalize 0-1
        print(f"Weather for {track}: Air {temp}°C, Track ~{track_temp}°C, Rain: {rain_prob}")
        return {'track_temp': track_temp, 'rain_prob': rain_prob}
    except Exception as e:
        print(f"Weather fetch failed: {e}")
        return {'track_temp': 30.0, 'rain_prob': 0.1}  # Fallback

def simulate_stint(base: float, slope: float, length: int, temp_factor: float = 1.0) -> float:
    """Simulate stint time: increasing laps."""
    times = [base + i * slope for i in range(1, length + 1)]
    return sum(times) * temp_factor

def simulate_strategy(track: str, team: str, weather: Dict, strategy: List[Tuple[str, int]]) -> Dict[str, any]:
    """Return total_time and dict with analytics (stint_times, deg_penalties, avg_lap_times, cum_times)."""
    total_laps = track_laps_dict[track]
    if sum(st[1] for st in strategy) != total_laps:
        raise ValueError(f"Strategy sums to {sum(st[1] for st in strategy)} != {total_laps} laps")
    
    rain_prob = weather['rain_prob']
    is_wet = rain_prob > 0.3
    temp = weather['track_temp']
    temp_factor = 1 + max(0, (temp - 30) / 100)
    
    # Track fallback: Average from all available if missing
    if track not in deg_models:
        print(f"No data for {track}; averaging from {len(deg_models)} tracks.")
        all_bases = {comp: [] for comp in ['SOFT', 'MEDIUM', 'HARD']}
        all_slopes = {comp: [] for comp in ['SOFT', 'MEDIUM', 'HARD']}
        for t in deg_models:
            for tm in deg_models[t]:
                for comp in ['SOFT', 'MEDIUM', 'HARD']:
                    if comp in deg_models[t][tm]:
                        all_bases[comp].append(deg_models[t][tm][comp]['base'])
                        all_slopes[comp].append(deg_models[t][tm][comp]['slope'])
        fallback = {comp: {'base': np.mean(all_bases[comp]), 'slope': np.mean(all_slopes[comp])} 
                    for comp in all_bases if all_bases[comp]}
        track_data = {'Other': fallback}  # Use 'Other' as default team here
    else:
        track_data = deg_models[track]
    
    # Team fallback: Average from track's teams if missing
    if team not in track_data:
        print(f"No data for {team} on {track}; averaging teams.")
        all_bases = {comp: [] for comp in ['SOFT', 'MEDIUM', 'HARD']}
        all_slopes = {comp: [] for comp in ['SOFT', 'MEDIUM', 'HARD']}
        for t_team in track_data:
            for comp in ['SOFT', 'MEDIUM', 'HARD']:
                if comp in track_data[t_team]:
                    all_bases[comp].append(track_data[t_team][comp]['base'])
                    all_slopes[comp].append(track_data[t_team][comp]['slope'])
        track_data[team] = {comp: {'base': np.mean(all_bases[comp]), 'slope': np.mean(all_slopes[comp])} 
                            for comp in all_bases if all_bases[comp]}
    
    total_time = 0.0
    analytics = {'stint_times': [], 'deg_penalties': [], 'avg_lap_times': [], 'cum_times': [0]}
    current_cum = 0.0
    prev_comp = None
    current_lap = 0
    for comp, stint_laps in strategy:
        orig_comp = comp
        if is_wet and comp != 'INTERMEDIATE':
            comp = 'INTERMEDIATE'
        
        # Handle INTER
        if comp == 'INTERMEDIATE':
            base = track_data[team]['HARD']['base'] + 3.0 if 'HARD' in track_data[team] else 95.0  # Ultimate fallback
            slope = track_data[team]['HARD']['slope'] * 1.5 if 'HARD' in track_data[team] else 0.03
        else:
            base = track_data[team][comp]['base']
            slope = track_data[team][comp]['slope']
        
        adjusted_slope = slope * temp_factor
        stint_times = [base + j * adjusted_slope for j in range(stint_laps)]
        stint_time = sum(stint_times)
        deg_penalty = sum(j * slope for j in range(stint_laps))  # Base slope for penalty
        avg_lap = stint_time / stint_laps
        
        # Build cum times
        for lap_time in stint_times:
            current_lap += 1
            current_cum += lap_time
            analytics['cum_times'].append(current_cum)
        
        total_time += stint_time
        analytics['stint_times'].append(stint_time)
        analytics['deg_penalties'].append(deg_penalty)
        analytics['avg_lap_times'].append(avg_lap)
        
        if prev_comp is not None:
            total_time += pit_time
            current_cum += pit_time
            analytics['cum_times'][-1] += pit_time  # Add pit to last lap of stint
        prev_comp = orig_comp
    
    if is_wet:
        total_time *= 1.1
    
    analytics['total_time'] = total_time
    return analytics

def optimize_strategy(track: str, team: str = 'Other', weather: Dict = None) -> Tuple[List[Tuple[str, int]], float, List[int], Dict]:
    """Enumerate common strategies, pick best."""
    total_laps = track_laps_dict[track]
    candidates = []
    
    # 1-stop strategies (exact sums)
    s1 = max(10, min(int(0.3 * total_laps), total_laps - 10))
    candidates.append([('SOFT', s1), ('HARD', total_laps - s1)])
    
    s2 = max(10, min(int(0.5 * total_laps), total_laps - 10))
    candidates.append([('MEDIUM', s2), ('HARD', total_laps - s2)])
    
    # 2-stop (exact sums)
    s3 = max(10, min(int(0.25 * total_laps), total_laps - 20))
    s4 = max(10, min(int(0.35 * total_laps), total_laps - s3 - 10))
    s5 = total_laps - s3 - s4
    candidates.append([('SOFT', s3), ('MEDIUM', s4), ('HARD', s5)])
    
    s6 = max(10, min(int(0.3 * total_laps), total_laps - 20))
    s7 = max(10, min(int(0.4 * total_laps), total_laps - s6 - 10))
    s8 = total_laps - s6 - s7
    candidates.append([('SOFT', s6), ('HARD', s7), ('HARD', s8)])
    
    s9 = max(10, min(int(0.3 * total_laps), total_laps - 20))
    s10 = max(10, min(int(0.3 * total_laps), total_laps - s9 - 10))
    s11 = total_laps - s9 - s10
    candidates.append([('MEDIUM', s9), ('SOFT', s10), ('HARD', s11)])
    
    # Aggressive soft-soft-hard
    s12 = max(10, min(int(0.2 * total_laps), total_laps - 20))
    s13 = max(10, min(int(0.3 * total_laps), total_laps - s12 - 10))
    s14 = total_laps - s12 - s13
    candidates.append([('SOFT', s12), ('SOFT', s13), ('HARD', s14)])
    
    if weather is None:
        weather = get_weather(track)
    
    best_time = float('inf')
    best_strat = None
    best_analytics = None
    for strat in candidates:
        try:
            sim_result = simulate_strategy(track, team, weather, strat)
            time = sim_result['total_time']
            if time < best_time:
                best_time = time
                best_strat = strat
                best_analytics = sim_result
        except Exception as e:
            print(f"Skipped strategy {strat}: {e}")
            continue
    
    # Safety: Fallback if none worked
    if best_strat is None:
        print("No strategy succeeded; using Medium one-stop fallback.")
        best_strat = [('MEDIUM', total_laps)]
        best_analytics = simulate_strategy(track, team, weather, best_strat)
        best_time = best_analytics['total_time']
    
    # Compute pit laps (cumsum of previous stints)
    pit_laps = [sum(s[1] for s in best_strat[:i]) for i in range(1, len(best_strat))]
    
    return best_strat, best_time, pit_laps, best_analytics

# Optional: LLM explanation (Groq)
try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def explain_strategy(strategy, track, weather):
        try:
            prompt = f"""
            You are an F1 strategy engineer. Analyze this strategy:
            Track: {track}
            Weather: {weather}
            Strategy: {strategy}
            Explain briefly how this helps optimize tire degradation, pit timing, and performance.
            """

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"LLM Error: {str(e)}. Check your GROQ_API_KEY."

except ImportError:
    def explain_strategy(*args):
        return "Groq SDK not installed. Run: pip install groq"

except Exception as e:
    # catch import-time errors (invalid key, API issues)
    def explain_strategy(*args, err=e):
        return f"Groq initialization error: {str(err)}"
