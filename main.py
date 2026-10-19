import uvicorn

from app.main import app

def main() -> None:
    """Run development server directly via python main.py."""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
