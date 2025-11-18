import fastf1 as ff1
from strategy_engine import optimize_strategy
import numpy as np
import pandas as pd
from weather_api import get_weather_from_openweather

SAMPLE_RACES = [
    (2021, 'Bahrain', 'R'), (2021, 'Monza', 'R'), (2021, 'Silverstone', 'R'), (2021, 'Abu Dhabi', 'R'),
    (2022, 'Bahrain', 'R'), (2022, 'Monza', 'R'), (2022, 'Silverstone', 'R'), (2022, 'Spa', 'R'),
    (2023, 'Bahrain', 'R'), (2023, 'Monza', 'R'), (2023, 'Silverstone', 'R'), (2023, 'Spa', 'R'),
    (2024, 'Bahrain', 'R'), (2024, 'Monza', 'R')
]

def get_actual_strategy(session):
    """Get winner's strategy: compounds & pit laps."""
    try:
        # Get winner from results
        results = session.results
        if results is None or len(results) == 0:
            return None, None, None
        winner_row = results[results['Position'] == 1]
        if winner_row.empty:
            return None, None, None
        winner_driver = winner_row['Abbreviation'].iloc[0]
        
        # Get winner's laps
        winner_laps = session.laps[session.laps['Driver'] == winner_driver]
        if winner_laps is None or len(winner_laps) == 0:
            return None, None, None
        
        # Group by stint to get strategy
        stints = winner_laps.groupby('Stint').agg({
            'Compound': 'first',
            'LapNumber': ['min', 'max', 'count']
        }).reset_index()
        
        stints.columns = ['Stint', 'Compound', 'StartLap', 'EndLap', 'LapCount']
        
        strategy = []
        pit_laps = []
        
        for _, row in stints.iterrows():
            comp = row['Compound']
            laps = int(row['LapCount'])
            strategy.append((comp, laps))
            
            # Pit laps are the start of each stint except the first
            if row['Stint'] > 1:
                pit_laps.append(int(row['StartLap']))
        
        # Calculate total time (approximate)
        total_lap_time = winner_laps['LapTime'].sum()
        if pd.isna(total_lap_time):
            return None, None, None
            
        actual_time = total_lap_time.total_seconds() + 22 * len(pit_laps)  # Add pit stop time
        
        return strategy, pit_laps, actual_time
        
    except Exception as e:
        print(f"    Strategy extraction error: {e}")
        return None, None, None

metrics = {'time_delta': [], 'strat_match': []}
successful_validations = 0

for year, track, event in SAMPLE_RACES:
    try:
        print(f"\nProcessing {year} {track}...")
        session = ff1.get_session(year, track, event)
        session.load()
        
        # Get weather data from OpenWeather API
        weather = get_weather_from_openweather(track)
        
        actual_strat, actual_pits, actual_time = get_actual_strategy(session)
        if actual_strat is None:
            print(f"  No winner data available")
            continue
        
        # Get prediction from strategy engine
        _, pred_time, pred_pits, _ = optimize_strategy(track, 'Other', weather)
        
        # Calculate metrics
        time_delta = abs(pred_time - actual_time) / actual_time * 100  # % error
        pit_match = 1 if len(pred_pits) == len(actual_pits) else 0
        
        metrics['time_delta'].append(time_delta)
        metrics['strat_match'].append(pit_match)
        successful_validations += 1
        
        print(f"  Actual: {len(actual_strat)} stints, {len(actual_pits)} pits, {actual_time/60:.1f}min")
        print(f"  Predicted: {len(pred_pits)+1} stints, {len(pred_pits)} pits, {pred_time/60:.1f}min")
        print(f"  Time error: {time_delta:.1f}%, Pit match: {'✓' if pit_match else '✗'}")
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

print(f"\n{'='*50}")
print(f"VALIDATION SUMMARY")
print(f"{'='*50}")

if metrics['time_delta']:
    avg_time_error = np.mean(metrics['time_delta'])
    pit_match_rate = np.mean(metrics['strat_match']) * 100
    
    print(f"Successful validations: {successful_validations}/{len(SAMPLE_RACES)}")
    print(f"Average time error: {avg_time_error:.1f}%")
    print(f"Pit strategy match rate: {pit_match_rate:.0f}%")
    
    summary = f"Validation Results: {successful_validations}/{len(SAMPLE_RACES)} races | Avg Time Error: {avg_time_error:.1f}% | Pit Match: {pit_match_rate:.0f}%"
else:
    print("No successful validations completed.")
    summary = "No successful validations completed."

# Save results for app
with open('val_results.txt', 'w', encoding='utf-8') as f:
    f.write("F1 Strategy Engine Validation Results\n")
    f.write("="*40 + "\n")
    f.write(f"Races tested: {len(SAMPLE_RACES)}\n")
    f.write(f"Successful validations: {successful_validations}\n")
    if metrics['time_delta']:
        f.write(f"Average time prediction error: {np.mean(metrics['time_delta']):.1f}%\n")
        f.write(f"Pit strategy match rate: {np.mean(metrics['strat_match'])*100:.0f}%\n")
    else:
        f.write("No successful validations completed.\n")
    f.write(f"\nNote: Weather data fetched from OpenWeather API\n")

print(f"\nResults saved to val_results.txt")