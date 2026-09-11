from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from src.db.database import Base
import datetime

class Match(Base):
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    tournament_name = Column(String)
    match_date = Column(DateTime, default=datetime.datetime.utcnow)
    location = Column(String)
    player_a = Column(String)
    player_b = Column(String)
    video_url = Column(String)
    
    # Relationships
    innings = relationship("Inning", back_populates="match", cascade="all, delete-orphan")

class Inning(Base):
    __tablename__ = "innings"
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    inning_number = Column(Integer)
    player_id = Column(String) # Player A or Player B
    
    # Relationships
    match = relationship("Match", back_populates="innings")
    shots = relationship("Shot", back_populates="inning", cascade="all, delete-orphan")

class Shot(Base):
    __tablename__ = "shots"
    
    id = Column(Integer, primary_key=True, index=True)
    inning_id = Column(Integer, ForeignKey("innings.id"))
    shot_number = Column(Integer)
    
    # Physics & Hit Data
    success = Column(Boolean, default=False)
    cue_speed = Column(Float, nullable=True)     # m/s
    thickness = Column(Float, nullable=True)     # Percentage (e.g., 50.0 for half)
    spin_x = Column(Float, nullable=True)        # Left/Right Tip
    spin_y = Column(Float, nullable=True)        # Top/Bottom Tip
    
    # Video References
    start_frame = Column(Integer, nullable=True)
    end_frame = Column(Integer, nullable=True)
    video_clip_path = Column(String, nullable=True)
    
    # Relationships
    inning = relationship("Inning", back_populates="shots")
