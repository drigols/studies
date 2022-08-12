from sqlalchemy import Column, INTEGER, String, TIMESTAMP, BIGINT, BOOLEAN, text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Product(Base):
    __tablename__ = "product"
    id = Column(INTEGER, primary_key=True)
    name = Column(String(1024), nullable=False)
    price = Column(BIGINT)
    is_available = (Column(BOOLEAN, default=True))
    seller_email = (Column(String(512), nullable=True))
    deleted = (Column(BOOLEAN, default=False))
    created_by = Column(INTEGER, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_by = Column(INTEGER, nullable=True)
    updated_at = Column(TIMESTAMP, nullable=True, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))
