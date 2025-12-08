from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from ..database import get_db
from ..models.user import User
from ..schemas import Token, RegisterForm, LoginForm, CurrentUser
from ..security import (
    create_access_token,
    verify_password,
    get_password_hash,
    ALGORITHM,
    SECRET_KEY,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ---------------------------
# Helpers
# ---------------------------
def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()


def get_current_user(token: str, db: Session = Depends(get_db)) -> User:
    """Extract user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
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


# ---------------------------
# Routes
# ---------------------------
@router.post("/signup", response_model=CurrentUser)
def register(form: RegisterForm, db: Session = Depends(get_db)):
    """Register a new user"""
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
def login(form_data: LoginForm, db: Session = Depends(get_db)):
    """Login with JSON {username, password}"""
    user = get_user_by_username(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": user.username},
        expires_delta=token_expires,
    )
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me", response_model=CurrentUser)
def me(current_user: User = Depends(get_current_user)):
    """Return current logged in user"""
    return current_user
