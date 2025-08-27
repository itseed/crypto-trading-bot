from flask import Flask, jsonify, request
import json
import multiprocessing
import yaml
import os
import bot
from typing import Union, Dict, Any, Tuple
from flask import Response

app = Flask(__name__)

process = None

@app.route("/status")
def status():
    global process
    try:
        if process and process.is_alive():
            try:
                if os.path.exists("paper_state.json"):
                    with open("paper_state.json", "r") as f:
                        state = json.load(f)
                    return jsonify({"status": "running", "state": state})
                else:
                    return jsonify({"status": "running", "state": "No state file found"})
            except Exception as e:
                return jsonify({"status": "running", "error": f"Error reading state: {str(e)}"}), 500
        else:
            return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"error": f"Error checking status: {str(e)}"}), 500

@app.route("/start", methods=['POST'])
def start():
    global process
    try:
        if process and process.is_alive():
            return jsonify({"error": "Bot is already running"}), 400
        
        process = multiprocessing.Process(target=bot.main)
        process.start()
        return jsonify({"status": "started", "pid": process.pid})
    except Exception as e:
        return jsonify({"error": f"Error starting bot: {str(e)}"}), 500

@app.route("/stop", methods=['POST'])
def stop():
    global process
    try:
        if process and process.is_alive():
            process.terminate()
            process.join(timeout=5)  # Wait up to 5 seconds for graceful termination
            if process.is_alive():
                process.kill()  # Force kill if still alive
            process = None
            return jsonify({"status": "stopped"})
        else:
            return jsonify({"error": "Bot is not running"}), 400
    except Exception as e:
        return jsonify({"error": f"Error stopping bot: {str(e)}"}), 500

@app.route("/config", methods=['GET', 'POST'])
def config() -> Union[Response, Tuple[Response, int]]:
    try:
        if request.method == 'GET':
            if os.path.exists("config.yaml"):
                with open("config.yaml", "r") as f:
                    config = yaml.safe_load(f)
                return jsonify(config)
            else:
                return jsonify({"error": "Config file not found"}), 404
        elif request.method == 'POST':
            # Validate the config data
            config_data = request.json
            if not config_data:
                return jsonify({"error": "No configuration data provided"}), 400
                
            with open("config.yaml", "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            return jsonify({"status": "updated"})
    except Exception as e:
        return jsonify({"error": f"Error handling config: {str(e)}"}), 500
    
    # Default return in case of unexpected execution path
    return jsonify({"error": "Unexpected execution path"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "crypto-trading-bot-web"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)