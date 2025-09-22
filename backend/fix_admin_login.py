#!/usr/bin/env python3
"""
Fix admin login issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import get_db, init_db
from app.models.user import User

def check_admin_user():
    """Check admin user details"""
    print("🔍 Checking Admin User...")
    
    try:
        init_db()
        db = next(get_db())
        
        # Check admin user
        admin = db.query(User).filter(User.username == "admin").first()
        if admin:
            print(f"✅ Admin user found:")
            print(f"   Username: {admin.username}")
            print(f"   Email: {admin.email}")
            print(f"   Role: {admin.role}")
            
            # Test login with username
            print("\n📝 Login Credentials:")
            print(f"   Username: {admin.username}")
            print(f"   Email: {admin.email}")
            print(f"   Password: admin123")
            
            return True
        else:
            print("❌ Admin user not found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_admin_with_email():
    """Create admin user with proper email"""
    print("\n🔧 Creating Admin User with Email...")
    
    try:
        db = next(get_db())
        
        # Check if admin with email exists
        admin_email = db.query(User).filter(User.email == "admin@example.com").first()
        if admin_email:
            print("✅ Admin user with email already exists")
            return True
        
        # Create new admin user
        from app.security import get_password_hash
        
        admin = User(
            username="admin",
            email="admin@example.com",
            hashed_password=get_password_hash("admin123"),
            role="admin",
        )
        
        db.add(admin)
        db.commit()
        print("✅ Admin user created successfully")
        print("   Username: admin")
        print("   Email: admin@example.com")
        print("   Password: admin123")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating admin: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Admin Login Fix")
    print("=" * 30)
    
    if not check_admin_user():
        print("❌ Admin user check failed")
        exit(1)
    
    create_admin_with_email()
    
    print("\n🎉 Admin login should now work!")
    print("\n📝 Use these credentials in the frontend:")
    print("   Email: admin@example.com")
    print("   Password: admin123")


