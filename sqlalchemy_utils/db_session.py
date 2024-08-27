from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import environ
from pathlib import Path

# Initialize environ
env = environ.Env()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Read .env file
environ.Env.read_env(BASE_DIR / '.env')

# engine = create_engine('sqlite:///db.sqlite3')
# SessionFactory = sessionmaker(bind=engine)
# Session = scoped_session(SessionFactory)

db_name = env('DB_NAME')
db_username = env('DB_USER')
db_host = env('DB_HOST')
db_port = env('DB_PORT')
db_password = env('DB_PASSWORD')

engine = create_engine(f'postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}')
SessionFactory = sessionmaker(bind=engine)
Session = scoped_session(SessionFactory)

def get_session():
    return Session()
