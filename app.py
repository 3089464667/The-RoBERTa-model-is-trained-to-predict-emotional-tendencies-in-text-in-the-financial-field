from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
import torch

model_name = 'updated4_sentiment_model'  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

app = Flask(__name__)
CORS(app)  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    sentiment_judgment = {
        0: [
            "The sentiment is negative. Expect potential market downturns or increased volatility.",
            "Negative outlook detected. Caution is advised as market turbulence could occur.",
            "Bearish sentiment observed. Possible market corrections or declines might happen.",
            "Pessimistic view noted. Market may face downward pressure or instability.",
            "Negative sentiment could lead to increased market risk or downturns.",
            "The market might encounter some challenges based on the current negative sentiment.",
            "Investors show caution as negative trends may affect market performance.",
            "A downtrend might be on the horizon given the prevailing negative sentiment.",
            "Market sentiment is gloomy, suggesting possible declines in asset values.",
            "Negative indicators are present, advising vigilance in market activities.",
            "Caution is warranted as negative sentiment prevails in the market.",
            "The market may experience headwinds due to the current negative sentiment.",
            "Investors should brace for possible market corrections amidst negative signals.",
            "Negative sentiment suggests potential challenges for market stability.",
            "The outlook is not promising, with negative sentiment influencing market trends.",
            "Expect potential setbacks as negative sentiment casts a shadow over the market.",
            "The market is under pressure, reflecting the current negative sentiment.",
            "Bearish trends are emerging as negative sentiment influences investor decisions.",
            "Negative sentiment could trigger risk-averse behavior among market participants.",
            "The market may struggle to gain ground amidst prevailing negative sentiment.",
            "Investors are wary as the negative sentiment suggests potential declines.",
            "The market outlook is bleak, with negative sentiment dominating discussions.",
            "Economic indicators align with the negative sentiment, hinting at downturns.",
            "Market forecasts are pessimistic, driven by the current negative sentiment.",
            "The investment climate is challenging, with negative sentiment prevailing.",
            "Market analysts predict challenges ahead, influenced by negative sentiment.",
            "Negative market sentiment could impact investor confidence and actions.",
            "The trading environment may become volatile due to negative sentiment.",
            "Market dynamics are expected to shift negatively, reflecting current sentiment.",
            "The market's negative sentiment may deter potential investors or actions."
        ],
        1: [
            "The sentiment is neutral. The market might remain stable with no significant changes.",
            "Neutral sentiment detected. Market conditions are likely to stay steady.",
            "Balanced outlook observed. Expect market stability with minor fluctuations.",
            "Market sentiment is neutral, suggesting a period of calm or equilibrium.",
            "Neutral view indicates steady market conditions, with little expected change.",
            "The market is in a holding pattern, with neutral sentiment prevailing.",
            "Investors can expect a stable market environment with the current neutral sentiment.",
            "Market conditions are balanced, reflecting the neutral sentiment among investors.",
            "Neutral sentiment suggests a wait-and-see approach in market activities.",
            "The market may experience minimal volatility in the current neutral sentiment.",
            "Market stability is anticipated as neutral sentiment persists.",
            "A steady market is expected, with neutral sentiment guiding investor behavior.",
            "The market is poised for a period of calm, with neutral sentiment dominating.",
            "Neutral sentiment points to a balanced market outlook with few surprises.",
            "Investors may find a stable environment as neutral sentiment takes hold.",
            "The market is likely to remain unchanged, reflecting the current neutral sentiment.",
            "Neutral sentiment suggests a cautious but stable market approach.",
            "Market trends are expected to be flat amidst prevailing neutral sentiment.",
            "The market is in equilibrium, with neutral sentiment balancing investor expectations.",
            "A quiet market period is expected, driven by the current neutral sentiment.",
            "Market forecasts indicate stability, with neutral sentiment at the forefront.",
            "The investment landscape is steady, supported by neutral sentiment.",
            "Market participants are adopting a neutral stance, reflecting current sentiment.",
            "The market's neutral sentiment suggests a period of observation and patience.",
            "Investors are maintaining a balanced approach amid neutral market sentiment.",
            "The market outlook is steady, with neutral sentiment influencing strategies.",
            "Neutral sentiment suggests a period of consolidation in the market.",
            "Market dynamics are expected to remain constant, reflecting neutral sentiment.",
            "The trading environment is calm, with neutral sentiment guiding actions.",
            "Market analysts predict stability, supported by the prevailing neutral sentiment."
        ],
        2: [
            "The sentiment is positive. Potential market upswings or opportunities may arise.",
            "Positive outlook detected. Market may experience growth or upward trends.",
            "Bullish sentiment observed. Opportunities for market gains or expansions exist.",
            "Optimistic view noted. Market could see positive movements or advancements.",
            "Positive sentiment could drive market growth or prosperity.",
            "Investors are hopeful as positive sentiment suggests potential market gains.",
            "The market may experience a rally, reflecting the current positive sentiment.",
            "Positive indicators are present, encouraging market participation and growth.",
            "The market is poised for potential gains, driven by the prevailing positive sentiment.",
            "Optimism among investors suggests potential upward momentum in the market.",
            "Positive sentiment is expected to bolster market confidence and activity.",
            "The market may benefit from a boost in investor sentiment, encouraging growth.",
            "Positive trends are emerging, suggesting a favorable market environment.",
            "Investors are optimistic about market prospects amid positive sentiment.",
            "The market outlook is bright, with positive sentiment supporting upward trends.",
            "Positive sentiment indicates potential opportunities for market expansion.",
            "The market is likely to gain strength, driven by the prevailing positive sentiment.",
            "Investor confidence is high, reflecting the positive sentiment in the market.",
            "The market may experience favorable conditions, supported by positive sentiment.",
            "A bullish market phase is anticipated, fueled by the current positive sentiment.",
            "Market forecasts are optimistic, driven by positive sentiment and trends.",
            "The investment landscape is vibrant, supported by positive sentiment.",
            "Market participants are adopting a positive stance, reflecting current sentiment.",
            "The market's positive sentiment suggests potential growth and opportunities.",
            "Investors are embracing a proactive approach amid positive market sentiment.",
            "The market outlook is encouraging, with positive sentiment influencing strategies.",
            "Positive sentiment suggests a period of expansion in the market.",
            "Market dynamics are expected to shift positively, reflecting current sentiment.",
            "The trading environment is dynamic, with positive sentiment driving actions.",
            "Market analysts predict growth, supported by the prevailing positive sentiment."
        ]
    }
    
    
    judgment = random.choice(sentiment_judgment[predicted_class_id])

    return jsonify({
        'predicted_class': predicted_class_id,
        'market_judgment': judgment
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
