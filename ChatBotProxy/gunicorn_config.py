from dotenv import load_dotenv, find_dotenv
import os

# Load the .env file
load_dotenv(find_dotenv(usecwd=True))

# Access environment variables
bind = f"{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '8000')}"
workers = int(os.getenv('WORKERS', 4))