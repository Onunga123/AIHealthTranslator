from datetime import timedelta
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

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ✅ Fix token URL (must be relative, not absolute)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

logger = logging.getLogger("auth")


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
    if db.query(User).filter((User.username == form.username) | (User.email == form.email)).first():
        raise HTTPException(status_code=400, detail="Username or email already registered")

    u = User(
        username=form.username,
        email=form.email,
        hashed_password=get_password_hash(form.password),
        role="user",  # default role
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    start = time.time()
    logger.info(f"[LOGIN] Attempt for username: {form_data.username}")
    user = get_user_by_username(db, form_data.username)
    logger.info(f"[LOGIN] DB query done in {time.time() - start:.3f}s")
    if not user:
        logger.warning("[LOGIN] User not found.")
        raise HTTPException(status_code=401, detail="Invalid username or password")

    pw_start = time.time()
    if not verify_password(form_data.password, user.hashed_password):
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


@router.get("/me", response_model=CurrentUser)
def me(current_user: User = Depends(get_current_user)):
    return current_user
