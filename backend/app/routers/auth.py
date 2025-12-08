from datetime import timedelta
import os
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import logging
import time

from ..database import get_db
from ..models.user import User
from ..schemas import Token, RegisterForm, CurrentUser
from ..security import (
    create_access_token,
    verify_password,
    get_password_hash,
    ALGORITHM,
    SECRET_KEY,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

from fastapi.responses import RedirectResponse, JSONResponse
from urllib.parse import urlencode
import requests
import uuid
from fastapi import Request
from sqlalchemy.exc import IntegrityError


router = APIRouter(prefix="/auth", tags=["Authentication"])

# Fix token URL (must be relative, not absolute)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Use propagate so logs appear in main.py handlers
logger = logging.getLogger("auth")
logger.propagate = True

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(db, username)
    if not user:
        raise credentials_exception
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user


@router.post("/signup", response_model=CurrentUser)
def register(form: RegisterForm, db: Session = Depends(get_db)):
    logger.info(f"[SIGNUP] Attempt for username: {form.username}")
    try:
        if db.query(User).filter((User.username == form.username) | (User.email == form.email)).first():
            logger.warning(f"[SIGNUP] Username or email already registered: {form.username} / {form.email}")
            raise HTTPException(status_code=400, detail="Username or email already registered")

        # Truncate password to 72 bytes for bcrypt
        truncated_password = form.password.encode('utf-8')[:72].decode('utf-8', errors='ignore')

        u = User(
            username=form.username,
            email=form.email,
            hashed_password=get_password_hash(truncated_password),
            role="user",  # default role
        )
        db.add(u)
        db.commit()
        db.refresh(u)
        logger.info(f"[SIGNUP] User successfully registered: {form.username}")
        return u
    except Exception as e:
        logger.error(f"[SIGNUP] Registration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    start = time.time()
    logger.info(f"[LOGIN] Attempt for username: {form_data.username}")
    try:
        user = get_user_by_username(db, form_data.username)
        logger.info(f"[LOGIN] DB query done in {time.time() - start:.3f}s")
        if not user:
            logger.warning("[LOGIN] User not found.")
            raise HTTPException(status_code=401, detail="Invalid username or password")

        # Truncate password for verification
        truncated_password = form_data.password.encode('utf-8')[:72].decode('utf-8', errors='ignore')
        pw_start = time.time()
        if not verify_password(truncated_password, user.hashed_password):
            logger.warning(f"[LOGIN] Password verification failed in {time.time() - pw_start:.3f}s")
            raise HTTPException(status_code=401, detail="Invalid username or password")
        logger.info(f"[LOGIN] Password verified in {time.time() - pw_start:.3f}s")

        token_start = time.time()
        token = create_access_token(
            data={"sub": user.username},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        logger.info(f"[LOGIN] Token created in {time.time() - token_start:.3f}s")
        logger.info(f"[LOGIN] Total login time: {time.time() - start:.3f}s")
        return {"access_token": token, "token_type": "bearer"}

    except Exception as e:
        logger.error(f"[LOGIN] Login failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Login failed")


@router.get("/me", response_model=CurrentUser)
def me(current_user: User = Depends(get_current_user)):
    logger.info(f"[ME] Current user fetched: {current_user.username}")
    return current_user



@router.get('/oauth/google')
def oauth_google_start():
    # Use environment variables to configure OAuth client
    client_id = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
    redirect_uri = os.getenv('GOOGLE_OAUTH_REDIRECT')
    if not client_id or not redirect_uri:
        return JSONResponse(status_code=501, content={"detail": "Google OAuth not configured. Set GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_REDIRECT."})

    params = {
        'client_id': client_id,
        'response_type': 'code',
        'scope': 'openid email profile',
        'redirect_uri': redirect_uri,
        'access_type': 'offline',
        'prompt': 'consent',
    }
    url = 'https://accounts.google.com/o/oauth2/v2/auth?' + urlencode(params)
    return RedirectResponse(url)


@router.get('/oauth/facebook')
def oauth_facebook_start():
    client_id = os.getenv('FACEBOOK_OAUTH_CLIENT_ID')
    redirect_uri = os.getenv('FACEBOOK_OAUTH_REDIRECT')
    if not client_id or not redirect_uri:
        return JSONResponse(status_code=501, content={"detail": "Facebook OAuth not configured. Set FACEBOOK_OAUTH_CLIENT_ID and FACEBOOK_OAUTH_REDIRECT."})

    params = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': 'email,public_profile',
        'response_type': 'code',
    }
    url = 'https://www.facebook.com/v12.0/dialog/oauth?' + urlencode(params)
    return RedirectResponse(url)



@router.get('/oauth/google/callback')
def oauth_google_callback(request: Request, code: str = None, db: Session = Depends(get_db)):
    if not code:
        return JSONResponse(status_code=400, content={"detail": "Missing code parameter"})

    client_id = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
    client_secret = os.getenv('GOOGLE_OAUTH_CLIENT_SECRET')
    redirect_uri = os.getenv('GOOGLE_OAUTH_REDIRECT')
    frontend_redirect = os.getenv('FRONTEND_REDIRECT', 'http://localhost:3000')

    if not client_id or not client_secret or not redirect_uri:
        return JSONResponse(status_code=501, content={"detail": "Google OAuth not configured. Set GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET and GOOGLE_OAUTH_REDIRECT."})

    token_url = 'https://oauth2.googleapis.com/token'
    data = {
        'code': code,
        'client_id': client_id,
        'client_secret': client_secret,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code',
    }
    try:
        tok_res = requests.post(token_url, data=data, timeout=10)
        tok_res.raise_for_status()
        tok = tok_res.json()
    except Exception as e:
        logger.error(f"[OAUTH-GOOGLE] Token exchange failed: {e}")
        return JSONResponse(status_code=502, content={"detail": "Token exchange failed"})

    # fetch profile
    try:
        id_token = tok.get('id_token')
        access_token = tok.get('access_token')
        profile_res = requests.get('https://www.googleapis.com/oauth2/v3/userinfo', headers={'Authorization': f'Bearer {access_token}'}, timeout=10)
        profile_res.raise_for_status()
        profile = profile_res.json()
    except Exception as e:
        logger.error(f"[OAUTH-GOOGLE] Failed to fetch profile: {e}")
        return JSONResponse(status_code=502, content={"detail": "Failed to fetch profile"})

    email = profile.get('email')
    username = profile.get('email').split('@')[0] if profile.get('email') else f'google_{uuid.uuid4().hex[:8]}'

    # find or create user
    user = db.query(User).filter((User.email == email) | (User.username == username)).first()
    if not user:
        try:
            user = User(username=username, email=email, hashed_password=get_password_hash(uuid.uuid4().hex), role='user')
            db.add(user)
            db.commit()
            db.refresh(user)
        except IntegrityError:
            db.rollback()
            user = db.query(User).filter(User.email == email).first()

    # issue JWT
    token = create_access_token(data={"sub": user.username})

    # redirect to frontend with token as fragment (safer than query)
    redirect_to = f"{frontend_redirect}/oauth_callback#access_token={token}"
    return RedirectResponse(redirect_to)


@router.get('/oauth/facebook/callback')
def oauth_facebook_callback(request: Request, code: str = None, db: Session = Depends(get_db)):
    if not code:
        return JSONResponse(status_code=400, content={"detail": "Missing code parameter"})

    client_id = os.getenv('FACEBOOK_OAUTH_CLIENT_ID')
    client_secret = os.getenv('FACEBOOK_OAUTH_CLIENT_SECRET')
    redirect_uri = os.getenv('FACEBOOK_OAUTH_REDIRECT')
    frontend_redirect = os.getenv('FRONTEND_REDIRECT', 'http://localhost:3000')

    if not client_id or not client_secret or not redirect_uri:
        return JSONResponse(status_code=501, content={"detail": "Facebook OAuth not configured. Set FACEBOOK_OAUTH_CLIENT_ID, FACEBOOK_OAUTH_CLIENT_SECRET and FACEBOOK_OAUTH_REDIRECT."})

    token_url = 'https://graph.facebook.com/v12.0/oauth/access_token'
    params = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'client_secret': client_secret,
        'code': code,
    }
    try:
        tok_res = requests.get(token_url, params=params, timeout=10)
        tok_res.raise_for_status()
        tok = tok_res.json()
    except Exception as e:
        logger.error(f"[OAUTH-FB] Token exchange failed: {e}")
        return JSONResponse(status_code=502, content={"detail": "Token exchange failed"})

    access_token = tok.get('access_token')
    try:
        profile_res = requests.get('https://graph.facebook.com/me', params={'fields': 'id,name,email', 'access_token': access_token}, timeout=10)
        profile_res.raise_for_status()
        profile = profile_res.json()
    except Exception as e:
        logger.error(f"[OAUTH-FB] Failed to fetch profile: {e}")
        return JSONResponse(status_code=502, content={"detail": "Failed to fetch profile"})

    email = profile.get('email')
    username = profile.get('name').replace(' ', '_') if profile.get('name') else f'fb_{uuid.uuid4().hex[:8]}'

    user = db.query(User).filter((User.email == email) | (User.username == username)).first()
    if not user:
        try:
            user = User(username=username, email=email, hashed_password=get_password_hash(uuid.uuid4().hex), role='user')
            db.add(user)
            db.commit()
            db.refresh(user)
        except IntegrityError:
            db.rollback()
            user = db.query(User).filter(User.email == email).first()

    token = create_access_token(data={"sub": user.username})
    redirect_to = f"{frontend_redirect}/oauth_callback#access_token={token}"
    return RedirectResponse(redirect_to)
