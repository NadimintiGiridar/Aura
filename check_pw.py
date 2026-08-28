import bcrypt
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://postgres:aura2024@localhost:5432/aura_db')
with engine.connect() as conn:
    result = conn.execute(text("SELECT email, password_hash FROM users WHERE email='sri@gmail.com'"))
    row = result.fetchone()
    if row:
        email, hashed = row[0], row[1]
        password = 'Sri@2005'
        pwd_bytes = password.encode('utf-8')[:72]
        hash_bytes = hashed.encode('utf-8')
        ok = bcrypt.checkpw(pwd_bytes, hash_bytes)
        print(f'Email: {email}')
        print(f'Password Sri@2005 matches: {ok}')
        ok2 = bcrypt.checkpw('sri@2005'.encode('utf-8')[:72], hash_bytes)
        print(f'Password sri@2005 (lower) matches: {ok2}')
    else:
        print('User not found!')
