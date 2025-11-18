import os
import anthropic
from typing import Optional, Dict, Any

class ClaudeAnalyzer:
    """
    Claude API integration for F1 strategy analysis and insights.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
        if not self.api_key:
            print("Warning: No Claude API key found. Set ANTHROPIC_API_KEY environment variable.")
            self.client = None
        else:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                print("Claude API client initialized successfully.")
            except Exception as e:
                print(f"Error initializing Claude client: {e}")
                self.client = None
    
    def analyze_strategy(self, strategy_data: Dict[str, Any]) -> Optional[str]:
        """
        Analyze F1 strategy using Claude.
        
        Args:
            strategy_data: Dictionary containing strategy information
            
        Returns:
            Claude's analysis as a string, or None if API unavailable
        """
        if not self.client:
            return "Claude API not available. Please set ANTHROPIC_API_KEY."
        
        try:
            # Format the strategy data for Claude
            prompt = self._format_strategy_prompt(strategy_data)
            
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            return f"Error analyzing strategy with Claude: {e}"
    
    def _format_strategy_prompt(self, data: Dict[str, Any]) -> str:
        """Format strategy data into a prompt for Claude."""
        
        track = data.get('track', 'Unknown')
        weather = data.get('weather', {})
        strategy = data.get('strategy', [])
        pit_laps = data.get('pit_laps', [])
        total_time = data.get('total_time', 0)
        
        prompt = f"""
As an F1 strategy expert, analyze this race strategy:

**Track:** {track}
**Weather Conditions:**
- Track Temperature: {weather.get('track_temp', 'N/A')}°C
- Rain Probability: {weather.get('rain_prob', 0)}

**Strategy:**
- Total Race Time: {total_time/60:.1f} minutes
- Number of Pit Stops: {len(pit_laps)}
- Pit Stop Laps: {pit_laps}
- Tire Strategy: {strategy}

Please provide:
1. **Strategy Assessment**: Is this an optimal strategy for the conditions?
2. **Key Insights**: What are the main strategic considerations?
3. **Alternative Approaches**: What other strategies could work?
4. **Risk Analysis**: What are the main risks with this strategy?
5. **Weather Impact**: How do the weather conditions affect this strategy?

Keep the analysis concise but insightful, focusing on the most important strategic elements.
"""
        
        return prompt
    
    def compare_strategies(self, actual_strategy: Dict, predicted_strategy: Dict) -> Optional[str]:
        """
        Compare actual vs predicted strategies using Claude.
        
        Args:
            actual_strategy: The actual race strategy used
            predicted_strategy: The AI-predicted strategy
            
        Returns:
            Claude's comparison analysis
        """
        if not self.client:
            return "Claude API not available."
        
        try:
            prompt = f"""
Compare these F1 strategies:

**ACTUAL STRATEGY (What happened in the race):**
- Pit Stops: {len(actual_strategy.get('pit_laps', []))}
- Pit Laps: {actual_strategy.get('pit_laps', [])}
- Total Time: {actual_strategy.get('total_time', 0)/60:.1f} min
- Tire Strategy: {actual_strategy.get('strategy', [])}

**PREDICTED STRATEGY (AI recommendation):**
- Pit Stops: {len(predicted_strategy.get('pit_laps', []))}
- Pit Laps: {predicted_strategy.get('pit_laps', [])}
- Total Time: {predicted_strategy.get('total_time', 0)/60:.1f} min
- Tire Strategy: {predicted_strategy.get('strategy', [])}

**Track:** {actual_strategy.get('track', 'Unknown')}

Analyze:
1. **Accuracy**: How close was the prediction?
2. **Key Differences**: What were the main strategic differences?
3. **Why the Difference**: What factors might explain the differences?
4. **Learning Points**: What can we learn to improve predictions?
5. **Context**: What race-specific factors might have influenced the actual strategy?

Provide a concise but thorough analysis.
"""
            
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            return f"Error comparing strategies: {e}"

def setup_claude_api():
    """
    Instructions for setting up Claude API.
    """
    print("To use Claude API integration:")
    print("1. Go to https://console.anthropic.com/")
    print("2. Sign up for an Anthropic account")
    print("3. Create an API key")
    print("4. Set it as environment variable: ANTHROPIC_API_KEY=your_key_here")
    print("5. Or add it to a .env file in your project")
    print("6. Install the anthropic package: pip install anthropic")
    print("\nNote: Claude API requires credits/billing to be set up.")

if __name__ == "__main__":
    # Test the Claude integration
    analyzer = ClaudeAnalyzer()
    
    if not analyzer.client:
        setup_claude_api()
    else:
        # Test with sample data
        test_data = {
            'track': 'Silverstone',
            'weather': {'track_temp': 18, 'rain_prob': 0.4},
            'strategy': [('MEDIUM', 25), ('HARD', 27)],
            'pit_laps': [25],
            'total_time': 5400  # 90 minutes
        }
        
        analysis = analyzer.analyze_strategy(test_data)
        print("Claude Analysis:")
        print(analysis)