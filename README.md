# AI-F1-strategy-maker

Visit at - https://dothrishi-ai-f1-strategy-maker-app-mqahah.streamlit.app/
 
An AI powered platform that suggests the optimal pit stop strategy based on trained data of past 140 races.

The model has been trained on race results and strategies of years - 2018, 2020, 2021, 2022, 2023 and 2024 using linear regression. 
Linear regression was implemented using following formula:-
lap_time=slope*number_of_laps+base_time

The y-axis represents the lap time and the x-axis represents lap in stint. 
The slop represents tire degradation for a particluar set of tires.

Our system fetches current weather from the track (eg - Saudi Arabia) through OpenWeather API and based on above calculations suggests the best strategies and win probability.

To provide a verification for predicted strategy, we have implemented a LLM feedback using groq API for gemini-2.5-flash model. The LLM provides detailed description of why the predicted strategy is accurate with its pros and cons. improvise this readme.md

<img src="screenshots/Screenshot 2025-11-18 150416.png">

<img src="screenshots/Screenshot 2025-11-18 150408.png">

<img src="screenshots/Screenshot 2025-11-18 150235.png">

<img src="screenshots/Screenshot 2025-11-18 150230.png">

<img src="screenshots/Screenshot 2025-11-18 150220.png">

<img src="screenshots/Screenshot 2025-11-18 150206.png">


