import os

class Settings:
    PROJECT_NAME: str = "Futuroscope HVAC & Visitor Optimization"
    VERSION: str = "1.0.0"
    
    # Chuỗi kết nối PostgreSQL (Thay đổi user, password, host, port, dbname phù hợp với DBeaver của bạn)
    # Cú pháp: postgresql://<username>:<password>@<host>:<port>/<dbname>
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:daibaohua@localhost:5432/futuroscope"
    )

settings = Settings()