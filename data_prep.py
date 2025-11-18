import fastf1 as ff1
from fastf1 import plotting
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os
from pathlib import Path

# Create cache dir if missing
os.makedirs('cache', exist_ok=True)

# Enable caching for FastF1
ff1.Cache.enable_cache('cache')

# Track-specific laps (approx; from FastF1)
TRACK_LAPS = {
    'Bahrain': 57, 'Saudi Arabia': 50, 'Australia': 58, 'Japan': 53, 'China': 56,
    'Miami': 57, 'Imola': 63, 'Monaco': 78, 'Canada': 70, 'Spain': 66,
    'Austria': 71, 'Silverstone': 52, 'Hungary': 70, 'Belgium': 44, 'Netherlands': 72,
    'Monza': 53, 'Azerbaijan': 51, 'Singapore': 62, 'USA': 56, 'Mexico': 71,
    'Brazil': 71, 'Las Vegas': 50, 'Qatar': 57, 'Abu Dhabi': 58
}

# Compounds map (FastF1 uses 'SOFT', etc.)
COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE']  # INTER for wet

# Teams (top ones; expand as needed)
TEAMS = ['Mercedes', 'Red Bull', 'Ferrari', 'McLaren', 'Aston Martin']

def fit_degradation(session, driver, compound):
    """Fit linear deg: lap_time = base + slope * lap_in_stint"""
    try:
        laps_df = session.laps.pick_driver(driver).pick_tires(compound)
        if len(laps_df) < 3:  # Relaxed: min 3 laps for fit
            print(f"  Insufficient laps ({len(laps_df)}) for {compound} on {driver}")
            return None, None
        laps_df['LapInStint'] = laps_df.groupby('Stint')['LapNumber'].transform(lambda x: x - x.min() + 1)
        mask = laps_df['LapInStint'] > 0  # Valid stints
        if mask.sum() < 3:
            print(f"  Insufficient valid stint laps ({mask.sum()}) for {compound} on {driver}")
            return None, None
        X = laps_df.loc[mask, 'LapInStint'].values.reshape(-1, 1)
        y = laps_df.loc[mask, 'LapTime'].dt.total_seconds().values
        reg = LinearRegression().fit(X, y)
        base = reg.intercept_
        slope = reg.coef_[0]  # seconds per lap deg
        print(f"    Fitted {compound}: base={base:.2f}s, slope={slope:.3f}s/lap")
        return base, slope
    except Exception as e:
        print(f"  Fit error for {compound} on {driver}: {e}")
        return None, None

def prepare_data():
    deg_data = {}  # {track: {team: {compound: (base_laptime, slope)}}}
    pit_time_avg = 22.0  # seconds; from historical avg
    processed_races = 0

    for year in range(2021, 2025):  # Recent years for better data
        for track in TRACK_LAPS:
            try:
                print(f"\n--- Processing {year} {track} ---")
                session = ff1.get_session(year, track, 'R')
                session.load()
                processed_races += 1
                weather = session.weather_data
                print(f"Weather: Max rainfall={weather['Rainfall'].max() if 'Rainfall' in weather.columns else 'N/A'}")
                if 'Rainfall' in weather.columns and weather['Rainfall'].max() > 1.0:  # Looser: Skip only heavy rain
                    print("  Skipped: Wet race")
                    continue
                print("  Dry race: Proceeding")
                avg_temp = weather['TrackTemp'].mean() if 'TrackTemp' in weather.columns else 30
                print(f"  Avg track temp: {avg_temp:.1f}°C")
                top_drivers_df = session.laps.pick_drivers('laps', 3)
                if top_drivers_df is None or len(top_drivers_df) == 0:
                    print("  No top drivers data")
                    continue
                top_drivers = top_drivers_df.drivers
                print(f"  Top drivers: {top_drivers}")
                for driver in top_drivers:
                    team = session.get_driver(driver)['TeamName']
                    if team not in TEAMS:
                        team = 'Other'
                    print(f"  Processing driver {driver} ({team})")
                    if track not in deg_data:
                        deg_data[track] = {}
                    if team not in deg_data[track]:
                        deg_data[track][team] = {}
                    for comp in COMPOUNDS[:-1]:  # Skip INTER for dry
                        base, slope = fit_degradation(session, driver, comp)
                        if base is not None:
                            # Adjust base for temp (simple: +0.01s per °C above 30)
                            temp_adjust = max(0, (avg_temp - 30) * 0.01)
                            base += temp_adjust
                            if comp not in deg_data[track][team]:
                                deg_data[track][team][comp] = {'base': [], 'slope': []}
                            deg_data[track][team][comp]['base'].append(base)
                            deg_data[track][team][comp]['slope'].append(slope)
                            print(f"    Added data for {comp}")
            except Exception as e:
                print(f"  Skipped {year} {track}: {e}")
                continue

    print(f"\n--- Summary: Processed {processed_races} races ---")
    print(f"Tracks with data before averaging: {list(deg_data.keys())}")

    # Average the fits (only if data exists)
    for track in list(deg_data.keys()):
        for team in deg_data[track]:
            for comp in list(deg_data[track][team].keys()):
                d = deg_data[track][team][comp]
                if len(d['base']) > 0:  # Only if data
                    deg_data[track][team][comp] = {
                        'base': np.mean(d['base']),
                        'slope': np.mean(d['slope'])
                    }
                    print(f"Averaged {track}/{team}/{comp}: base={deg_data[track][team][comp]['base']:.2f}s, slope={deg_data[track][team][comp]['slope']:.3f}s/lap")
                else:
                    del deg_data[track][team][comp]  # Clean empty

    # If still empty, add hardcoded Pirelli averages (for testing; remove for production)
    if not deg_data:
        print("No data fitted; adding hardcoded averages.")
        hardcoded = {
            'Bahrain': {'Other': {'SOFT': {'base': 92.5, 'slope': 0.045}, 'MEDIUM': {'base': 93.8, 'slope': 0.025}, 'HARD': {'base': 95.2, 'slope': 0.015}}},
            'China': {'Other': {'SOFT': {'base': 94.0, 'slope': 0.040}, 'MEDIUM': {'base': 95.3, 'slope': 0.022}, 'HARD': {'base': 96.7, 'slope': 0.012}}},
            'Monza': {'Other': {'SOFT': {'base': 85.2, 'slope': 0.060}, 'MEDIUM': {'base': 86.5, 'slope': 0.035}, 'HARD': {'base': 88.0, 'slope': 0.020}}},
            # Add similar for all tracks to ensure coverage
            'Silverstone': {'Other': {'SOFT': {'base': 89.1, 'slope': 0.055}, 'MEDIUM': {'base': 90.4, 'slope': 0.030}, 'HARD': {'base': 91.8, 'slope': 0.018}}},
            'Monaco': {'Other': {'SOFT': {'base': 81.2, 'slope': 0.035}, 'MEDIUM': {'base': 82.5, 'slope': 0.020}, 'HARD': {'base': 84.0, 'slope': 0.010}}},
            # ... (abbrev; expand with real Pirelli data for all 24)
        }
        deg_data = hardcoded
        for track in deg_data:
            for team in deg_data[track]:
                for comp in deg_data[track][team]:
                    print(f"Hardcoded {track}/{team}/{comp}: base={deg_data[track][team][comp]['base']:.2f}s, slope={deg_data[track][team][comp]['slope']:.3f}s/lap")

    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(deg_data, 'models/deg_models.pkl')
    joblib.dump(pit_time_avg, 'models/pit_time.pkl')
    joblib.dump(TRACK_LAPS, 'models/track_laps.pkl')
    print(f"Data prepared! Final tracks: {list(deg_data.keys())}")

if __name__ == "__main__":
    prepare_data()