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
    video_path = Column(String)

    # Which cue ball each player owns, read from the colour of their scoreboard
    # row. Without it a score cannot be tied to the ball seen moving.
    player_a_ball = Column(String)  # "white" or "yellow"
    player_b_ball = Column(String)

    fps = Column(Float)
    mm_per_px = Column(Float)
    calibration_error_mm = Column(Float)
    # Trajectories live in a .npz beside the video rather than in the database:
    # a play holds a few thousand floats, and the point of collecting them is to
    # read them back in bulk.
    track_path = Column(String)
    track_start_second = Column(Float)

    # What the scoreboard says the match contained, against what was extracted.
    # A match is only as useful as the share of its plays the broadcast showed.
    expected_plays = Column(Integer)
    detected_plays = Column(Integer)

    innings = relationship("Inning", back_populates="match", cascade="all, delete-orphan")


class Inning(Base):
    __tablename__ = "innings"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    inning_number = Column(Integer)
    player_id = Column(String)
    cue_ball = Column(String)

    # From the scoreboard: the points made in this turn, hence how many plays it
    # must hold. Recording it lets a later pass tell a complete inning from one
    # the broadcast only partly showed.
    points = Column(Integer)
    expected_plays = Column(Integer)
    complete = Column(Boolean, default=False)

    match = relationship("Match", back_populates="innings")
    shots = relationship("Shot", back_populates="inning", cascade="all, delete-orphan")


class Shot(Base):
    """One play: a layout, a stroke, and whether it scored."""

    __tablename__ = "shots"

    id = Column(Integer, primary_key=True, index=True)
    inning_id = Column(Integer, ForeignKey("innings.id"))
    shot_number = Column(Integer)
    cue_ball = Column(String)

    success = Column(Boolean, default=False)
    # Where the verdict came from. The scoreboard cannot time a point to a play
    # reliably - it moves anywhere from fourteen seconds early to forty-four
    # late - so the trajectory decides, and the board is the cross-check.
    verdict_source = Column(String)  # "trajectory" | "scoreboard" | "both"
    scoreboard_success = Column(Boolean, nullable=True)
    cushions_before_second = Column(Integer, nullable=True)
    first_object_ball = Column(String, nullable=True)
    second_object_ball = Column(String, nullable=True)

    # The layout played from, in table millimetres, origin at the corner of the
    # cushion nose line. This is the question every stored play is an answer to.
    white_x = Column(Float, nullable=True)
    white_y = Column(Float, nullable=True)
    yellow_x = Column(Float, nullable=True)
    yellow_y = Column(Float, nullable=True)
    red_x = Column(Float, nullable=True)
    red_y = Column(Float, nullable=True)

    # Where the balls finished. The layout a play leaves behind is the layout
    # the next one is played from, so a run of shots is a chain and this is the
    # link between them.
    end_white_x = Column(Float, nullable=True)
    end_white_y = Column(Float, nullable=True)
    end_yellow_x = Column(Float, nullable=True)
    end_yellow_y = Column(Float, nullable=True)
    end_red_x = Column(Float, nullable=True)
    end_red_y = Column(Float, nullable=True)

    cue_speed = Column(Float, nullable=True)  # m/s over the first frames of travel
    cue_travel_mm = Column(Float, nullable=True)
    thickness = Column(Float, nullable=True)
    spin_x = Column(Float, nullable=True)
    spin_y = Column(Float, nullable=True)

    start_frame = Column(Integer, nullable=True)  # index into the match's track file
    end_frame = Column(Integer, nullable=True)
    start_second = Column(Float, nullable=True)  # position in the source video
    end_second = Column(Float, nullable=True)
    video_clip_path = Column(String, nullable=True)

    # Why a play may not be fit for the dataset. Kept rather than filtered out
    # at write time, so the same extraction can be re-filtered later.
    complete = Column(Boolean, default=True)  # balls were seen coming to rest
    inferred = Column(Boolean, default=False)  # strike never on screen
    is_replay = Column(Boolean, default=False)
    label_confirmed = Column(Boolean, default=False)
    usable = Column(Boolean, default=False)
    rejected_for = Column(String, nullable=True)

    inning = relationship("Inning", back_populates="shots")
