import os
from dotenv import load_dotenv
import openai
# Load environment variables from .env file




if __name__ == '__main__':
    print('Hello, OpenAI!')
    load_dotenv()
    # Access environment variables
    SECRET_KEY = os.getenv("deepseekAPI")
    DEBUG = os.getenv("DEBUG")
    print(SECRET_KEY)
    
       
    