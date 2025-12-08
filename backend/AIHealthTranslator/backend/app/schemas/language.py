from pydantic import BaseModel

class LanguageOut(BaseModel):
    id: int
    name: str
    code: str

    class Config:
        orm_mode = True
