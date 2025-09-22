from sqlalchemy.orm import Session
from ..database import engine
from ..models.dictionary_term import DictionaryTerm

def seed_dictionary_terms():
    terms = [
        {"term_en": "malaria", "term_sw": "malaria", "term_luo": "malaria"},
        {"term_en": "fever", "term_sw": "homa", "term_luo": "hero"},
        {"term_en": "headache", "term_sw": "maumivu ya kichwa", "term_luo": "rembe"},
    ]
    with Session(engine) as db:
        for t in terms:
            exists = db.query(DictionaryTerm).filter(DictionaryTerm.term_en == t["term_en"]).first()
            if not exists:
                term = DictionaryTerm(**t)
                db.add(term)
        db.commit()
