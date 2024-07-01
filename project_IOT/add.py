pip install fastapi uvicorn sqlalchemy

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel

# กำหนด URL ของฐานข้อมูล SQLite
DATABASE_URL = "sqlite:///./test.db"

# สร้างเซสชันอัตโนมัติ
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# สร้าง Base class สำหรับการสร้าง Table ในฐานข้อมูล
Base = declarative_base()

# สร้างโมเดล Pydantic เพื่อรับข้อมูล
class Item(BaseModel):
    id: int
    name: str
    description: str

# สร้าง Table ในฐานข้อมูล
class ItemTable(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)

# สร้าง FastAPI app
app = FastAPI()

# เรียกใช้งาน SessionLocal
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# สร้าง API endpoint สำหรับดึงข้อมูลจากฐานข้อมูล
@app.get("/items/{item_id}", response_model=Item)
def read_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ItemTable).filter(ItemTable.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

