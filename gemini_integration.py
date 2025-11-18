import os
import google.generativeai as genai
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiAnalyzer:
    """
    Gemini API integration for F1 strategy analysis and insights.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            print("Warning: No Gemini API key found. Set GEMINI_API_KEY environment variable.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                print("Gemini API client initialized successfully.")
            except Exception as e:
                print(f"Error initializing Gemini client: {e}")
                self.model = None
    
    def analyze_strategy(self, strategy_data: Dict[str, Any]) -> Optional[str]:
        """
        Analyze F1 strategy using Gemini.
        
        Args:
            strategy_data: Dictionary containing strategy information
            
        Returns:
            Gemini's analysis as a string, or None if API unavailable
        """
        if not self.model:
            return "Gemini API not available. Please set GEMINI_API_KEY."
        
        try:
            # Format the strategy data for Gemini
            prompt = self._format_strategy_prompt(strategy_data)
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error analyzing strategy with Gemini: {e}"
    
    def _format_strategy_prompt(self, data: Dict[str, Any]) -> str:
        """Format strategy data into a prompt for Gemini."""
        
        track = data.get('track', 'Unknown')
        weather = data.get('weather', {})
        strategy = data.get('strategy', [])
        pit_laps = data.get('pit_laps', [])
        total_time = data.get('total_time', 0)
        
        prompt = f"""
You are an expert F1 race strategist. Analyze this race strategy:

**Track:** {track}
**Weather Conditions:**
- Track Temperature: {weather.get('track_temp', 'N/A')}°C
- Rain Probability: {weather.get('rain_prob', 0)}

**Strategy Details:**
- Total Race Time: {total_time/60:.1f} minutes
- Number of Pit Stops: {len(pit_laps)}
- Pit Stop Laps: {pit_laps}
- Tire Strategy: {strategy}

Please provide a comprehensive analysis covering:

1. **Strategy Assessment**: Is this an optimal strategy for the given conditions?
2. **Key Strategic Insights**: What are the main considerations that make this strategy work?
3. **Alternative Approaches**: What other viable strategies could have been used?
4. **Risk Analysis**: What are the potential risks and how to mitigate them?
5. **Weather Impact**: How do the weather conditions specifically affect this strategy?
6. **Track-Specific Factors**: What makes this strategy suitable for {track}?

Keep the analysis detailed but concise, focusing on actionable insights that could improve future strategy decisions.
"""
        
        return prompt
    
    def compare_strategies(self, actual_strategy: Dict, predicted_strategy: Dict) -> Optional[str]:
        """
        Compare actual vs predicted strategies using Gemini.
        
        Args:
            actual_strategy: The actual race strategy used
            predicted_strategy: The AI-predicted strategy
            
        Returns:
            Gemini's comparison analysis
        """
        if not self.model:
            return "Gemini API not available."
        
        try:
            prompt = f"""
As an F1 strategy expert, compare these two strategies and provide insights:

**ACTUAL STRATEGY (What happened in the race):**
- Track: {actual_strategy.get('track', 'Unknown')}
- Pit Stops: {len(actual_strategy.get('pit_laps', []))}
- Pit Stop Laps: {actual_strategy.get('pit_laps', [])}
- Total Race Time: {actual_strategy.get('total_time', 0)/60:.1f} minutes
- Tire Compounds Used: {actual_strategy.get('strategy', [])}

**PREDICTED STRATEGY (AI recommendation):**
- Pit Stops: {len(predicted_strategy.get('pit_laps', []))}
- Pit Stop Laps: {predicted_strategy.get('pit_laps', [])}
- Predicted Time: {predicted_strategy.get('total_time', 0)/60:.1f} minutes
- Recommended Tires: {predicted_strategy.get('strategy', [])}

**Weather Context:**
- Track Temperature: {actual_strategy.get('weather', {}).get('track_temp', 'N/A')}°C
- Rain Conditions: {actual_strategy.get('weather', {}).get('rain_prob', 0)}

Please analyze:

1. **Prediction Accuracy**: How accurate was the AI prediction compared to reality?
2. **Strategic Differences**: What were the key differences between predicted and actual strategies?
3. **Performance Gap**: What explains the time difference between predicted and actual?
4. **Context Factors**: What race-day factors (safety cars, incidents, tire degradation) might have influenced the actual strategy?
5. **Model Improvements**: Based on this comparison, what could improve future predictions?
6. **Strategic Lessons**: What strategic insights can we learn from this comparison?

Provide actionable insights that could help improve the AI model's future predictions.
"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error comparing strategies with Gemini: {e}"
    
    def generate_race_insights(self, validation_results: Dict[str, Any]) -> Optional[str]:
        """
        Generate insights from validation results using Gemini.
        
        Args:
            validation_results: Dictionary containing validation metrics
            
        Returns:
            Gemini's insights on the validation results
        """
        if not self.model:
            return "Gemini API not available."
        
        try:
            successful_races = validation_results.get('successful_validations', 0)
            total_races = validation_results.get('total_races', 0)
            avg_time_error = validation_results.get('avg_time_error', 0)
            pit_match_rate = validation_results.get('pit_match_rate', 0)
            
            prompt = f"""
As an F1 data analyst, interpret these validation results for an AI strategy prediction model:

**Validation Summary:**
- Races Successfully Analyzed: {successful_races}/{total_races}
- Average Time Prediction Error: {avg_time_error:.1f}%
- Pit Strategy Match Rate: {pit_match_rate:.0f}%

**Performance Analysis Needed:**
1. **Overall Model Performance**: How good are these results for F1 strategy prediction?
2. **Strengths**: What does the model do well?
3. **Weaknesses**: Where does the model need improvement?
4. **Benchmarking**: How do these results compare to typical F1 strategy prediction accuracy?
5. **Improvement Recommendations**: What specific areas should be focused on for better predictions?
6. **Data Quality**: What might be affecting the prediction accuracy?

Please provide a comprehensive assessment that would help improve the AI model's performance.
"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating insights with Gemini: {e}"

def setup_gemini_api():
    """
    Instructions for setting up Gemini API.
    """
    print("Gemini API Setup Instructions:")
    print("1. Go to https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Create a new API key")
    print("4. Set it as environment variable: GEMINI_API_KEY=your_key_here")
    print("5. Or add it to your .env file")
    print("6. Install the required package: pip install google-generativeai python-dotenv")
    print("\nNote: Gemini API has a generous free tier!")

if __name__ == "__main__":
    # Test the Gemini integration
    analyzer = GeminiAnalyzer()
    
    if not analyzer.model:
        setup_gemini_api()
    else:
        # Test with sample data
        test_data = {
            'track': 'Silverstone',
            'weather': {'track_temp': 18, 'rain_prob': 0.4},
            'strategy': [('MEDIUM', 25), ('HARD', 27)],
            'pit_laps': [25],
            'total_time': 5400  # 90 minutes
        }
        
        print("Testing Gemini integration...")
        analysis = analyzer.analyze_strategy(test_data)
        print("Gemini Analysis:")
        print(analysis)