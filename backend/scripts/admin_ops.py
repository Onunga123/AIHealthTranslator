import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # add backend/ to path

from app.database import SessionLocal  # type: ignore
from app.models.user import User  # type: ignore


def promote_user(email: str | None, username: str | None) -> dict:
    session = SessionLocal()
    try:
        user: User | None = None
        if email:
            user = session.query(User).filter(User.email == email).first()
        if user is None and username:
            user = session.query(User).filter(User.username == username).first()

        if user is None:
            return {"status": "not_found"}

        user.role = "admin"
        session.commit()
        session.refresh(user)
        return {
            "status": "ok",
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
        }
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Admin operations")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_promote = sub.add_parser("promote", help="Promote a user to admin")
    p_promote.add_argument("--email", type=str, default=None)
    p_promote.add_argument("--username", type=str, default=None)

    args = parser.parse_args()

    if args.cmd == "promote":
        result = promote_user(args.email, args.username)
        print(result)


if __name__ == "__main__":
    main()



