from sqlalchemy import (
    Column, 
    Integer, 
    BigInteger, 
    String, 
    Float, 
    Date, 
    DateTime, 
    ForeignKey, 
    Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# ==============================================================================
# 1. CÁC BẢNG DIMENSIONS (CHIỀU DỮ LIỆU)
# ==============================================================================

class DimAttraction(Base):
    """Bảng lưu danh mục trò chơi/công trình."""
    __tablename__ = "dim_attraction"

    id_attraction = Column(String(10), primary_key=True)
    name = Column(String(100), nullable=False)

    # Relationships (Nối tới các bảng con)
    cadence = relationship("DimCadence", back_populates="attraction", uselist=False, cascade="all, delete-orphan")
    facts_attraction = relationship("FactAttractionHourly", back_populates="attraction", cascade="all, delete-orphan")
    facts_elec = relationship("FactElecHourly", back_populates="attraction", cascade="all, delete-orphan")
    facts_ec = relationship("FactECHourly", back_populates="attraction", cascade="all, delete-orphan")


class DimCadence(Base):
    """Bảng tĩnh thông số kỹ thuật & diện tích công trình."""
    __tablename__ = "dim_cadence"

    id_attraction = Column(String(10), ForeignKey("dim_attraction.id_attraction", ondelete="CASCADE"), primary_key=True)
    surface = Column(Float, nullable=True)
    capacite_salle = Column(Float, nullable=True)
    capacite_file_attente = Column(Float, nullable=True)
    capacite_pre_salle = Column(Float, nullable=True)
    duree_longue = Column(Float, nullable=True)
    duree_courte = Column(Float, nullable=True)
    cycle_max_vl = Column(Float, nullable=True)
    duty_cycle_max_vl = Column(Float, nullable=True)
    cycle_max_vc = Column(Float, nullable=True)
    duty_cycle_max_vc = Column(Float, nullable=True)

    attraction = relationship("DimAttraction", back_populates="cadence")


class DimTemps(Base):
    """Bảng thời gian cấp GIỜ (YYYYMMDDHH)."""
    __tablename__ = "dim_temps"

    temps_id = Column(BigInteger, primary_key=True)
    datetime = Column(DateTime, nullable=False)
    date = Column(Date, nullable=False)
    annee = Column(Integer, nullable=False)
    mois = Column(Integer, nullable=False)
    jour = Column(Integer, nullable=False)
    heure = Column(Integer, nullable=False)
    jour_semaine = Column(Integer, nullable=False)
    is_weekend = Column(Integer, nullable=False)
    jf = Column(Integer, default=0)

    # Relationships
    weather = relationship("DimWeather", back_populates="temps", uselist=False, cascade="all, delete-orphan")
    facts_attraction = relationship("FactAttractionHourly", back_populates="temps", cascade="all, delete-orphan")
    facts_elec = relationship("FactElecHourly", back_populates="temps", cascade="all, delete-orphan")
    facts_ec = relationship("FactECHourly", back_populates="temps", cascade="all, delete-orphan")


class DimHoraire(Base):
    """Bảng lịch hoạt động công viên (cấp NGÀY)."""
    __tablename__ = "dim_horaire"

    datetime = Column(DateTime, primary_key=True)
    date = Column(Date, nullable=True)
    is_open = Column(Integer, nullable=True)
    h_ouv = Column(String(20), nullable=True)
    h_ferm = Column(String(20), nullable=True)
    frequentation = Column(Integer, nullable=True)
    type_frequentation = Column(String(20), nullable=True)
    jf = Column(Integer, nullable=True)


class DimWeather(Base):
    """Bảng thời tiết cấp GIỜ."""
    __tablename__ = "dim_weather"

    temps_id = Column(BigInteger, ForeignKey("dim_temps.temps_id", ondelete="CASCADE"), primary_key=True)
    datetime = Column(DateTime, nullable=False)
    date_key = Column(Date, nullable=True)
    heure = Column(Integer, nullable=True)
    temperature = Column(Float, nullable=True)
    humidite = Column(Float, nullable=True)
    rayonnement_solaire = Column(Float, nullable=True)
    day_degree_cold = Column(Float, nullable=True)
    day_degree_hot = Column(Float, nullable=True)
    temp_max = Column(Float, nullable=True)
    temp_min = Column(Float, nullable=True)
    temp_moy = Column(Float, nullable=True)
    humidite_max = Column(Float, nullable=True)
    humidite_min = Column(Float, nullable=True)
    humidite_moy = Column(Float, nullable=True)

    temps = relationship("DimTemps", back_populates="weather")


# ==============================================================================
# 2. CÁC BẢNG FACT (SỰ KIỆN / ĐO LƯỜNG)
# ==============================================================================

class FactAttractionHourly(Base):
    """Bảng lượt khách và trạng thái trò chơi cấp GIỜ."""
    __tablename__ = "fact_attraction_hourly"

    fact_id = Column(BigInteger, primary_key=True, autoincrement=True)
    temps_id = Column(BigInteger, ForeignKey("dim_temps.temps_id", ondelete="CASCADE"), nullable=False, index=True)
    id_attraction = Column(String(10), ForeignKey("dim_attraction.id_attraction", ondelete="CASCADE"), nullable=False, index=True)
    datetime = Column(DateTime, nullable=False)
    visitor_count = Column(Integer, default=0)
    ouvert = Column(Float, default=0)
    interrompu = Column(Float, default=0)
    operation = Column(Float, default=0)

    temps = relationship("DimTemps", back_populates="facts_attraction")
    attraction = relationship("DimAttraction", back_populates="facts_attraction")


class FactElecHourly(Base):
    """Bảng dữ liệu tiêu thụ điện năng cấp GIỜ."""
    __tablename__ = "fact_elec_hourly"

    fact_id = Column(BigInteger, primary_key=True, autoincrement=True)
    temps_id = Column(BigInteger, ForeignKey("dim_temps.temps_id", ondelete="CASCADE"), nullable=False, index=True)
    id_attraction = Column(String(10), ForeignKey("dim_attraction.id_attraction", ondelete="CASCADE"), nullable=False, index=True)
    datetime = Column(DateTime, nullable=False)
    metric_name = Column(String(50), nullable=False, index=True)
    value = Column(Float, default=0.0)

    temps = relationship("DimTemps", back_populates="facts_elec")
    attraction = relationship("DimAttraction", back_populates="facts_elec")


class FactECHourly(Base):
    """Bảng dữ liệu tiêu thụ nhiệt (Eau Chaude) cấp GIỜ."""
    __tablename__ = "fact_ec_hourly"

    fact_id = Column(BigInteger, primary_key=True, autoincrement=True)
    temps_id = Column(BigInteger, ForeignKey("dim_temps.temps_id", ondelete="CASCADE"), nullable=False, index=True)
    id_attraction = Column(String(10), ForeignKey("dim_attraction.id_attraction", ondelete="CASCADE"), nullable=False, index=True)
    datetime = Column(DateTime, nullable=False)
    metric_name = Column(String(50), nullable=False, index=True)
    value = Column(Float, default=0.0)

    temps = relationship("DimTemps", back_populates="facts_ec")
    attraction = relationship("DimAttraction", back_populates="facts_ec")