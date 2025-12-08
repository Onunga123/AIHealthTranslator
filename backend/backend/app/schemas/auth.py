from pydantic import BaseModel, EmailStr

class RegisterForm(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginForm(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CurrentUser(BaseModel):
    id: int
    username: str
    email: EmailStr
    role: str

    model_config = {"from_attributes": True}
