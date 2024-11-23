async function getSentiment(text) {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    });

    const result = await response.json();
    console.log('Market judgment:', result.predicted_class);

    document.getElementById('result').innerText = 'Predicted Sentiment Class: ' + result.market_judgment;
}

document.getElementById('submitButton').addEventListener('click', () => {
    const userInput = document.getElementById('textInput').value;
    getSentiment(userInput);
});
