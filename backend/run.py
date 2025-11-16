import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from api.app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)