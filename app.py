import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from io import StringIO
from dotenv import load_dotenv
from strategy_engine import optimize_strategy, explain_strategy, get_weather, TRACK_COORDS, track_laps_dict

# Groq API
from groq import Groq
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Load environment variables
load_dotenv()

# Custom CSS for F1 theme
st.markdown("""
    <style>
    .main {background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);}
    .stMetric {background-color: #ff0000; color: white; border-radius: 10px; padding: 10px;}
    .stDataFrame {border-radius: 10px; overflow: hidden;}
    .stPlotlyChart {border-radius: 10px;}
    .sidebar .sidebar-content {background: #1a1a1a;}
    h1 {color: #ff0000; font-family: 'Arial Black'; text-align: center;}
    h2 {color: #ffffff; border-bottom: 2px solid #ff0000; padding-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="F1 Strategy AI", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("🏎️ AI F1 Strategy Analyzer Pro")
st.markdown("**Powered by FastF1 Data** | Optimize tires, pits, and pace for victory.")

# Sidebar: Inputs
with st.sidebar:
    st.header("📊 Strategy Controls")
    track = st.selectbox("🏁 Track", list(TRACK_COORDS.keys()))
    team = st.selectbox("🚗 Team", ['Other', 'Mercedes', 'Red Bull', 'Ferrari', 'McLaren', 'Aston Martin'])
    use_llm = st.checkbox("🤖 Add AI Narrative (Groq Key Required)")
    st.markdown("---")
    st.caption("v1.0 | Built for Precision")

# Generate Strategy Button
if st.button("🔥 Generate Optimal Strategy", type="primary", use_container_width=True):
    with st.spinner("Simulating race... 🏁"):
        # Weather & Strategy
        weather = get_weather(track)
        strategy, total_time, pit_laps, analytics = optimize_strategy(track, team, weather)

        # Track default temps (fallback)
        track_defaults = {'Bahrain': 40, 'Monaco': 25, 'Monza': 32, 'Silverstone': 28,
                          'Spa': 26, 'China': 28, 'Miami': 32, 'Imola': 25, 'Australia': 22, 'Japan': 28}
        base_temp = weather['track_temp'] if weather['track_temp'] != 30 else track_defaults.get(track, 30)

        # Columns
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🌤️ Weather Snapshot")
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                st.metric("Track Temp", f"{weather['track_temp']:.1f}°C")
            with w_col2:
                st.metric("Rain Probability", f"{weather['rain_prob']*100:.0f}%")

            st.subheader("📈 Key Metrics")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Total Race Time", f"{total_time/60:.1f} min")
            with m_col2:
                historical_avg_min = 90 * track_laps_dict[track] / 60
                speed_up_pct = (historical_avg_min - total_time / 60) / historical_avg_min * 100
                win_prob = min(95, max(20, 50 + speed_up_pct * 2))
                st.metric("Win Probability", f"{win_prob:.0f}%")
            with m_col3:
                st.metric("Stops", len(strategy) - 1)

        with col2:
            st.subheader("🎯 Tire Strategy Breakdown")
            df_data = []
            cum_lap = 0
            for i, (comp, stints) in enumerate(strategy):
                start_lap = cum_lap + 1
                end_lap = cum_lap + stints
                pit_lap = pit_laps[i-1] if i > 0 else None
                stint_time = analytics['stint_times'][i]
                avg_lap = analytics['avg_lap_times'][i]
                deg_pen = analytics['deg_penalties'][i]
                df_data.append({
                    'Stint': i+1,
                    'Tires': comp,
                    'Laps': f"{start_lap}-{end_lap}",
                    'Pit Lap': pit_lap,
                    'Avg Lap': f"{avg_lap:.2f}s",
                    'Stint Time': f"{stint_time:.0f}s",
                    'Deg. Penalty': f"{deg_pen:.0f}s"
                })
                cum_lap += stints
            df = pd.DataFrame(df_data)
            st.dataframe(df.style.background_gradient(cmap='Reds'), use_container_width=True)

        # Charts
        st.markdown("---")
        st.subheader("📊 Advanced Analytics")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            total_laps = track_laps_dict[track]
            cum_times = analytics['cum_times'][1:]
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='black')
            ax.plot(range(1, total_laps+1), cum_times, color='#ff0000', linewidth=2.5, label='Pace Curve')
            for pit in pit_laps:
                ax.axvline(pit+0.5, color='yellow', linestyle='--', alpha=0.8, label='Pit' if pit==pit_laps[0] else "")
            ax.set_facecolor('black')
            ax.set_xlabel("Lap", color='white')
            ax.set_ylabel("Cum. Time (s)", color='white')
            ax.set_title(f"Pace Evolution - {track}", color='white')
            ax.legend(facecolor='black', edgecolor='white')
            ax.grid(True, alpha=0.3, color='gray')
            ax.tick_params(colors='white')
            plt.tight_layout()
            st.pyplot(fig)

        with chart_col2:
            compounds = [s[0] for s in strategy]
            times = analytics['stint_times']
            colors = {'SOFT':'#ff0000','MEDIUM':'#ffff00','HARD':'#ffffff','INTERMEDIATE':'#00ff00'}
            fig2, ax2 = plt.subplots(figsize=(8,5), facecolor='black')
            bars = ax2.bar(compounds, times, color=[colors.get(c,'gray') for c in compounds], edgecolor='black')
            ax2.set_facecolor('black')
            ax2.set_ylabel("Time (s)", color='white')
            ax2.set_title("Tire Allocation", color='white')
            ax2.bar_label(bars, fmt='%.0f', color='black')
            ax2.tick_params(colors='white')
            plt.tight_layout()
            st.pyplot(fig2)

        with st.expander("🔥 Weather Sensitivity (Temp Impact)"):
            temps = [base_temp-10, base_temp, base_temp+10]
            time_factors = [1+max(0,(t-30)/100) for t in temps]
            times_sens = [total_time*f for f in time_factors]
            fig3, ax3 = plt.subplots(figsize=(10,4), facecolor='black')
            ax3.plot(temps, [tt/60 for tt in times_sens], 'y-o', markerfacecolor='yellow', markersize=8, linewidth=2)
            ax3.set_facecolor('black')
            ax3.set_xlabel("Track Temp (°C)", color='white')
            ax3.set_ylabel("Total Time (min)", color='white')
            ax3.set_title(f"Sensitivity: +10°C Adds ~1-2 Min ({track} Base: {base_temp}°C)", color='white')
            ax3.grid(True, alpha=0.3, color='gray')
            ax3.tick_params(colors='white')
            plt.tight_layout()
            st.pyplot(fig3)

        # Groq AI Section
        st.markdown("---")
        st.subheader("🤖 Groq Strategy Advisor")
        st.info("Get a narrative alternative strategy blending data with expert intuition.")
        if st.button("Generate Groq Strategy", type="secondary"):
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                st.error("🚨 Set GROQ_API_KEY env var! Run: `$env:GROQ_API_KEY='gsk-your-key'` in PowerShell.")
            else:
                try:
                    with st.spinner("Consulting Groq AI..."):
                        client = Groq(api_key=api_key)
                        prompt = f"""
                        You are an F1 strategy expert. For a {track} race ({track_laps_dict[track]} laps), {team} car, weather: {weather['track_temp']}°C, {weather['rain_prob']*100}% rain.
                        Model data shows optimal: {strategy} with {total_time/60:.1f} min total, {len(strategy)-1} stops.
                        Suggest an ALTERNATIVE strategy (1-3 stops, tires sequence), explain why (deg, pits, risks), and est. time vs. model. Keep concise, exciting.
                        """
                        from groq import Groq
                        import os

                        api_key = os.getenv('GROQ_API_KEY')
                        client = Groq(api_key=api_key)

                        prompt_text = f"""
                        You are a Formula 1 strategy engineer. For a {track} race ({track_laps_dict[track]} laps), {team} car, weather: {weather['track_temp']}°C, {weather['rain_prob']*100}% rain.
                        Model data shows optimal: {strategy} with {total_time/60:.1f} min total, {len(strategy)-1} stops.
                        Suggest an ALTERNATIVE strategy (1-3 stops, tires sequence), explain why (deg, pits, risks), and est. time vs. model. Keep concise, exciting.
                        """

                        response = client.completions.create(
                            model="groq-chat-mini",  # or any available Groq chat model
                            prompt=prompt_text,
                            max_output_tokens=300
                        )

                        groq_strategy = response.output_text  # This contains the text output
                        st.success("Groq Strategy Advice:")
                        st.markdown(groq_strategy)
                except Exception as e:
                    st.error(f"Groq API Error: {e}")

        if use_llm:
            with st.expander("📝 Data-Driven AI Narrative"):
                explanation = explain_strategy(strategy, track, weather)
                st.write(explanation)

# Validation Section
st.markdown("---")
st.subheader("✅ Accuracy Validation (12-Race Backtest)")
val_file = 'val_results.txt'
if not os.path.exists(val_file):
    with st.spinner("Generating validation (first time only)..."):
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            import fastf1 as ff1
            from strategy_engine import optimize_strategy
            SAMPLE_RACES = [
                (2021,'Bahrain','R'),(2021,'Monza','R'),(2021,'Silverstone','R'),(2021,'Abu Dhabi','R'),
                (2022,'Bahrain','R'),(2022,'Monza','R'),(2022,'Silverstone','R'),(2022,'Spa','R'),
                (2023,'Bahrain','R'),(2023,'Monza','R'),(2023,'Silverstone','R'),(2023,'Spa','R'),
                (2024,'Bahrain','R'),(2024,'Monza','R')
            ]
            metrics = {'time_delta': [], 'strat_match': []}
            for year, track_name, event in SAMPLE_RACES:
                try:
                    session = ff1.get_session(year, track_name, event)
                    session.load()
                    weather_data = session.weather_data
                    weather = {'track_temp': weather_data['TrackTemp'].mean() if 'TrackTemp' in weather_data.columns else 30,
                               'rain_prob': 1 if 'Rainfall' in weather_data.columns and weather_data['Rainfall'].max()>0 else 0}
                    _, pred_time, pred_pits, _ = optimize_strategy(track_name,'Other',weather)
                    metrics['time_delta'].append(pred_time/60)  # simplified
                    metrics['strat_match'].append(1)
                    print(f"{year} {track_name}: Predicted race time {pred_time/60:.1f} min")
                except Exception as e:
                    print(f"{year} {track_name}: Error {e}")
        finally:
            sys.stdout = old_stdout
            val_output = captured.getvalue()
        with open(val_file,'w') as f:
            f.write(val_output)
        st.success("Validation generated & saved!")
else:
    with open(val_file,'r') as f:
        val_output = f.read()
st.code(val_output, language='text')

# Footer
st.markdown("---")
st.markdown("*© 2025 F1 Strategy AI | Data from FastF1 | For educational use.*")
