#!/usr/bin/env python3
"""
Simple login test
"""

import requests
import json

def test_login():
    """Test admin login"""
    print("🔍 Testing Admin Login...")
    
    base_url = "http://localhost:8000"
    
    # Test login with admin credentials
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    
    try:
        print(f"Attempting login to {base_url}/auth/login")
        print(f"Credentials: {login_data}")
        
        response = requests.post(f"{base_url}/auth/login", data=login_data)
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response body: {response.text}")
        
        if response.status_code == 200:
            token_data = response.json()
            print("✅ Login successful!")
            print(f"Token: {token_data['access_token'][:20]}...")
            
            # Test getting user info
            headers = {"Authorization": f"Bearer {token_data['access_token']}"}
            user_response = requests.get(f"{base_url}/auth/me", headers=headers)
            
            if user_response.status_code == 200:
                user_data = user_response.json()
                print("✅ User info retrieved!")
                print(f"Username: {user_data['username']}")
                print(f"Role: {user_data['role']}")
            else:
                print(f"❌ Failed to get user info: {user_response.text}")
                
        else:
            print("❌ Login failed")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_login()


