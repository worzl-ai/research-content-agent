"""
Research & Content Agent - Foundation Agent
"""
import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'agent': 'research-content-agent',
        'status': 'operational',
        'version': '1.0.0',
        'capabilities': [
            'keyword_research',
            'market_analysis', 
            'content_generation',
            'competitor_analysis'
        ]
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'agent': 'research-content-agent',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
