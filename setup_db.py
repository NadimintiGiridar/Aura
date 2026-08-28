"""
AURA Database Setup Script
Creates the aura_db database and all tables with corrected URL
"""
import os, sys

# Force correct DB URL before any imports
os.environ['DATABASE_URL'] = 'postgresql://postgres:admin@localhost:5432/aura_db'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text

DB_URL = 'postgresql://postgres:aura2024@localhost:5432/aura_db'
POSTGRES_URL = 'postgresql://postgres:aura2024@localhost:5432/postgres'

# Step 1: Create database if needed
print('Step 1: Connecting to PostgreSQL...')
try:
    engine = create_engine(POSTGRES_URL, isolation_level='AUTOCOMMIT')
    with engine.connect() as conn:
        row = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = 'aura_db'")).fetchone()
        if not row:
            conn.execute(text('CREATE DATABASE aura_db'))
            print('  Created database: aura_db')
        else:
            print('  Database aura_db already exists')
    engine.dispose()
except Exception as e:
    print(f'  Error creating database: {e}')
    print('  Trying to continue anyway...')

# Step 2: Create tables
print('Step 2: Creating tables...')
try:
    from app.database.base import Base
    from app.models import user, conversation, message, document  # noqa - registers models

    engine2 = create_engine(DB_URL)
    Base.metadata.create_all(bind=engine2)
    engine2.dispose()

    print('  SUCCESS! Tables created:')
    print('  - users')
    print('  - conversations')
    print('  - messages')
    print('  - documents')
    print()
    print('Database setup complete! You can now start the backend.')
except Exception as e:
    print(f'  ERROR creating tables: {e}')
    sys.exit(1)
