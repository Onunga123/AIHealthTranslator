#!/usr/bin/env python3
"""
Test backend connection and identify timeout issues
"""

import requests
import time

def test_backend_connection():
    """Test if backend is accessible"""
    print("🔍 Testing Backend Connection...")
    
    # Test different URLs
    urls = [
        "http://localhost:8000/",
        "http://127.0.0.1:8000/",
        "http://0.0.0.0:8000/"
    ]
    
    for url in urls:
        print(f"\nTesting: {url}")
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            end_time = time.time()
            
            print(f"✅ Success! Status: {response.status_code}")
            print(f"   Response time: {end_time - start_time:.2f} seconds")
            print(f"   Response: {response.text[:100]}...")
            
            # Test auth endpoint
            auth_url = url.replace("/", "/auth/login")
            print(f"\nTesting auth endpoint: {auth_url}")
            
            auth_response = requests.post(auth_url, 
                data={"username": "admin", "password": "admin123"}, 
                timeout=10)
            print(f"   Auth response: {auth_response.status_code}")
            print(f"   Auth body: {auth_response.text[:100]}...")
            
            return True
            
        except requests.exceptions.Timeout:
            print(f"❌ Timeout after 10 seconds")
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection refused")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return False

def test_frontend_backend_communication():
    """Test if frontend can reach backend"""
    print("\n🔍 Testing Frontend-Backend Communication...")
    
    # Simulate frontend request
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    
    try:
        print("Testing login request...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/auth/login",
            data=login_data,
            headers=headers,
            timeout=30
        )
        
        end_time = time.time()
        print(f"✅ Login request completed in {end_time - start_time:.2f} seconds")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 200:
            token_data = response.json()
            print(f"   Token received: {token_data['access_token'][:20]}...")
            
            # Test authenticated request
            auth_headers = {
                "Authorization": f"Bearer {token_data['access_token']}",
                "Accept": "application/json"
            }
            
            me_response = requests.get(
                "http://localhost:8000/auth/me",
                headers=auth_headers,
                timeout=10
            )
            
            print(f"   User info: {me_response.status_code}")
            print(f"   User data: {me_response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Login request timed out")
    except Exception as e:
        print(f"❌ Login request failed: {e}")

def check_ports():
    """Check what's running on common ports"""
    print("\n🔍 Checking Ports...")
    
    import subprocess
    try:
        result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        ports = ['8000', '3000', '5173', '5174', '5175']
        for port in ports:
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    print(f"✅ Port {port} is listening: {line.strip()}")
                    break
            else:
                print(f"❌ Port {port} is not listening")
                
    except Exception as e:
        print(f"❌ Error checking ports: {e}")

if __name__ == "__main__":
    print("🚀 Backend Connection Test")
    print("=" * 40)
    
    check_ports()
    
    if test_backend_connection():
        test_frontend_backend_communication()
    else:
        print("\n❌ Backend is not accessible!")
        print("\n📝 Troubleshooting tips:")
        print("1. Make sure the backend server is running in VSCode")
        print("2. Check if the server is running on port 8000")
        print("3. Check VSCode terminal for any error messages")
        print("4. Try restarting the backend server")


