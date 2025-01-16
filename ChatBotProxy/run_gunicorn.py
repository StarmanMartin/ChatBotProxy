import os
from dotenv import load_dotenv
from gunicorn.app.base import BaseApplication
from ChatBotProxy.app import app  # Import your Flask app

# Load environment variables from .env
load_dotenv()

class GunicornApp(BaseApplication):
    def __init__(self, application, options=None):
        self.application = application
        self.options = options or {}
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def run():
    options = {
        "bind": f"{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '8000')}",
        "workers": int(os.getenv("WORKERS", 4)),
    }
    GunicornApp(app, options).run()


if __name__ == '__main__':
    run()