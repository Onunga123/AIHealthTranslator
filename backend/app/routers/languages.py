from fastapi import APIRouter

router = APIRouter(prefix="/languages", tags=["Languages"])

SUPPORTED = [
    {"code": "en", "name": "English"},
    {"code": "sw", "name": "Kiswahili"},
    {"code": "luo", "name": "Luo (Dholuo)"},
]

@router.get("/supported")
def get_supported_languages():
    return {"languages": SUPPORTED}
