"""
Test Suite for AURA Authentication System (Phase 7)
Uses asyncio + httpx.AsyncClient with ASGITransport for bulletproof FastAPI testing.
"""
import sys
import os
import unittest
import asyncio
import httpx
from fastapi import FastAPI

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.api.auth import router as auth_router
from app.database.connection import SessionLocal, create_all_tables
from app.models.user import User

app = FastAPI()
app.include_router(auth_router)


class TestAuthSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_all_tables()
        cls.test_email = "nadimintigiridar13@gmail.com"
        cls.test_name = "Giridar13"
        cls.test_pass = "AuraSecurePassword123!"
        cls.cleanup_test_user(cls.test_email)

    @classmethod
    def tearDownClass(cls):
        cls.cleanup_test_user(cls.test_email)

    @classmethod
    def cleanup_test_user(cls, email: str):
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email.lower().strip()).first()
            if user:
                db.delete(user)
                db.commit()
        finally:
            db.close()

    def test_01_registration_success(self):
        print("\n--> Running test_01_registration_success...")
        async def _run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                payload = {
                    "full_name": self.test_name,
                    "email": self.test_email,
                    "password": self.test_pass,
                    "confirm_password": self.test_pass
                }
                res = await client.post("/auth/register", json=payload)
                self.assertEqual(res.status_code, 201, f"Expected 201, got {res.status_code}: {res.text}")
                data = res.json()
                self.assertIn("access_token", data)
                self.assertEqual(data["token_type"], "bearer")
                self.assertEqual(data["user"]["full_name"], self.test_name)
                self.assertEqual(data["user"]["email"], self.test_email)

        asyncio.run(_run())

        # Check database persistence and password hashing
        db = SessionLocal()
        try:
            db_user = db.query(User).filter(User.email == self.test_email).first()
            self.assertIsNotNone(db_user)
            self.assertEqual(db_user.full_name, self.test_name)
            self.assertNotEqual(db_user.password_hash, self.test_pass)  # Password must NOT be plaintext
            self.assertGreater(len(db_user.password_hash), 20)
            print("    [PASS] User created in PostgreSQL with hashed password.")
        finally:
            db.close()

    def test_02_registration_duplicate_email_409(self):
        print("--> Running test_02_registration_duplicate_email_409...")
        async def _run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                payload = {
                    "full_name": self.test_name,
                    "email": self.test_email,
                    "password": self.test_pass
                }
                res = await client.post("/auth/register", json=payload)
                self.assertEqual(res.status_code, 409)
                self.assertEqual(res.json()["detail"], "An account with this email already exists.")

        asyncio.run(_run())
        print("    [PASS] 409 Conflict returned with correct error message.")

    def test_03_login_and_jwt_auth(self):
        print("--> Running test_03_login_and_jwt_auth...")
        async def _run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                payload = {
                    "email": self.test_email,
                    "password": self.test_pass
                }
                res = await client.post("/auth/login", json=payload)
                self.assertEqual(res.status_code, 200)

                data = res.json()
                self.assertIn("access_token", data)
                token = data["access_token"]

                # Access /auth/me with JWT token
                me_res = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
                self.assertEqual(me_res.status_code, 200)
                me_data = me_res.json()
                self.assertEqual(me_data["email"], self.test_email)
                self.assertEqual(me_data["full_name"], self.test_name)

        asyncio.run(_run())
        print("    [PASS] Login successful & JWT auth validated.")

    def test_04_validation_errors_422(self):
        print("--> Running test_04_validation_errors_422...")
        async def _run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                # Short name
                res1 = await client.post("/auth/register", json={"full_name": "A", "email": "valid@example.com", "password": "password123"})
                self.assertEqual(res1.status_code, 422)

                # Invalid email format
                res2 = await client.post("/auth/register", json={"full_name": "Valid Name", "email": "invalid-email", "password": "password123"})
                self.assertEqual(res2.status_code, 422)

                # Short password
                res3 = await client.post("/auth/register", json={"full_name": "Valid Name", "email": "valid@example.com", "password": "short"})
                self.assertEqual(res3.status_code, 422)

        asyncio.run(_run())
        print("    [PASS] Validation errors correctly returned 422.")


if __name__ == "__main__":
    unittest.main()
