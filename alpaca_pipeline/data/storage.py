"""Data storage module for persisting market data."""

import os
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..config import get_config
from ..utils import get_logger, timing
from .models import BarData, TradeData, QuoteData, AssetInfo, PortfolioSnapshot

logger = get_logger(__name__)

Base = declarative_base()


class BarDataTable(Base):
    """SQLAlchemy model for bar data."""
    __tablename__ = 'bars'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    trade_count = Column(Integer)
    vwap = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<BarData(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"


class TradeDataTable(Base):
    """SQLAlchemy model for trade data."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    size = Column(Integer, nullable=False)
    conditions = Column(Text)
    exchange = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)


class QuoteDataTable(Base):
    """SQLAlchemy model for quote data."""
    __tablename__ = 'quotes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    bid_price = Column(Float, nullable=False)
    bid_size = Column(Integer, nullable=False)
    ask_price = Column(Float, nullable=False)
    ask_size = Column(Integer, nullable=False)
    bid_exchange = Column(String(10))
    ask_exchange = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)


class AssetInfoTable(Base):
    """SQLAlchemy model for asset information."""
    __tablename__ = 'assets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    asset_class = Column(String(50), nullable=False)
    exchange = Column(String(50), nullable=False)
    tradable = Column(Boolean, nullable=False)
    marginable = Column(Boolean, nullable=False)
    shortable = Column(Boolean, nullable=False)
    easy_to_borrow = Column(Boolean, nullable=False)
    fractionable = Column(Boolean, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PortfolioSnapshotTable(Base):
    """SQLAlchemy model for portfolio snapshots."""
    __tablename__ = 'portfolio_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    equity = Column(Float, nullable=False)
    buying_power = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    portfolio_value = Column(Float, nullable=False)
    long_market_value = Column(Float, nullable=False)
    short_market_value = Column(Float, nullable=False)
    day_trade_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DataStorage:
    """Data storage manager for market data persistence."""
    
    def __init__(self, database_url: Optional[str] = None, data_path: Optional[str] = None):
        """Initialize data storage.
        
        Args:
            database_url: Database connection URL
            data_path: Path for file-based storage
        """
        self.config = get_config()
        self.database_url = database_url or self.config.database_url
        self.data_path = Path(data_path or self.config.data_storage_path)
        
        # Create data directory if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        
        logger.info(
            "DataStorage initialized",
            database_url=self.database_url,
            data_path=str(self.data_path)
        )
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    @timing()
    def store_bars(
        self,
        bars: List[BarData],
        timeframe: str,
        batch_size: int = 1000
    ) -> int:
        """Store bar data to database.
        
        Args:
            bars: List of BarData objects
            timeframe: Data timeframe
            batch_size: Batch size for bulk insert
            
        Returns:
            Number of records stored
        """
        if not bars:
            return 0
        
        session = self.get_session()
        stored_count = 0
        
        try:
            # Process in batches
            for i in range(0, len(bars), batch_size):
                batch = bars[i:i + batch_size]
                
                # Convert to database records
                db_records = []
                for bar in batch:
                    db_record = BarDataTable(
                        symbol=bar.symbol,
                        timeframe=timeframe,
                        timestamp=bar.timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        trade_count=bar.trade_count,
                        vwap=bar.vwap
                    )
                    db_records.append(db_record)
                
                # Use upsert for PostgreSQL or INSERT OR REPLACE for SQLite
                if 'postgresql' in self.database_url:
                    self._upsert_bars_postgresql(session, db_records)
                else:
                    self._upsert_bars_sqlite(session, db_records)
                
                stored_count += len(batch)
                
                logger.debug(
                    "Stored bar batch",
                    batch_size=len(batch),
                    total_stored=stored_count
                )
            
            session.commit()
            
            logger.info(
                "Bars stored successfully",
                count=stored_count,
                timeframe=timeframe,
                symbols=list(set(bar.symbol for bar in bars))
            )
            
            return stored_count
            
        except Exception as e:
            session.rollback()
            logger.error(
                "Failed to store bars",
                error=str(e),
                count=len(bars)
            )
            raise
        finally:
            session.close()
    
    def _upsert_bars_postgresql(self, session: Session, records: List[BarDataTable]):
        """Upsert bars for PostgreSQL."""
        for record in records:
            stmt = pg_insert(BarDataTable).values(
                symbol=record.symbol,
                timeframe=record.timeframe,
                timestamp=record.timestamp,
                open=record.open,
                high=record.high,
                low=record.low,
                close=record.close,
                volume=record.volume,
                trade_count=record.trade_count,
                vwap=record.vwap
            )
            
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'timeframe', 'timestamp'],
                set_={
                    'open': stmt.excluded.open,
                    'high': stmt.excluded.high,
                    'low': stmt.excluded.low,
                    'close': stmt.excluded.close,
                    'volume': stmt.excluded.volume,
                    'trade_count': stmt.excluded.trade_count,
                    'vwap': stmt.excluded.vwap
                }
            )
            
            session.execute(stmt)
    
    def _upsert_bars_sqlite(self, session: Session, records: List[BarDataTable]):
        """Upsert bars for SQLite."""
        for record in records:
            stmt = sqlite_insert(BarDataTable).values(
                symbol=record.symbol,
                timeframe=record.timeframe,
                timestamp=record.timestamp,
                open=record.open,
                high=record.high,
                low=record.low,
                close=record.close,
                volume=record.volume,
                trade_count=record.trade_count,
                vwap=record.vwap
            )
            
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'timeframe', 'timestamp'],
                set_={
                    'open': stmt.excluded.open,
                    'high': stmt.excluded.high,
                    'low': stmt.excluded.low,
                    'close': stmt.excluded.close,
                    'volume': stmt.excluded.volume,
                    'trade_count': stmt.excluded.trade_count,
                    'vwap': stmt.excluded.vwap
                }
            )
            
            session.execute(stmt)
    
    @timing()
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[BarData]:
        """Retrieve bar data from database.
        
        Args:
            symbol: Symbol to retrieve
            timeframe: Data timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum number of records
            
        Returns:
            List of BarData objects
        """
        session = self.get_session()
        
        try:
            query = session.query(BarDataTable).filter(
                BarDataTable.symbol == symbol,
                BarDataTable.timeframe == timeframe
            )
            
            if start:
                query = query.filter(BarDataTable.timestamp >= start)
            if end:
                query = query.filter(BarDataTable.timestamp <= end)
            
            query = query.order_by(BarDataTable.timestamp)
            
            if limit:
                query = query.limit(limit)
            
            db_records = query.all()
            
            # Convert to BarData objects
            bars = []
            for record in db_records:
                bar = BarData(
                    symbol=record.symbol,
                    timestamp=record.timestamp,
                    open=record.open,
                    high=record.high,
                    low=record.low,
                    close=record.close,
                    volume=record.volume,
                    trade_count=record.trade_count,
                    vwap=record.vwap
                )
                bars.append(bar)
            
            logger.debug(
                "Retrieved bars from database",
                symbol=symbol,
                timeframe=timeframe,
                count=len(bars)
            )
            
            return bars
            
        except Exception as e:
            logger.error(
                "Failed to retrieve bars",
                symbol=symbol,
                timeframe=timeframe,
                error=str(e)
            )
            raise
        finally:
            session.close()
    
    @timing()
    def store_assets(self, assets: List[AssetInfo]) -> int:
        """Store asset information.
        
        Args:
            assets: List of AssetInfo objects
            
        Returns:
            Number of records stored
        """
        if not assets:
            return 0
        
        session = self.get_session()
        stored_count = 0
        
        try:
            for asset in assets:
                db_record = AssetInfoTable(
                    symbol=asset.symbol,
                    name=asset.name,
                    asset_class=asset.asset_class,
                    exchange=asset.exchange,
                    tradable=asset.tradable,
                    marginable=asset.marginable,
                    shortable=asset.shortable,
                    easy_to_borrow=asset.easy_to_borrow,
                    fractionable=asset.fractionable
                )
                
                # Use merge for upsert
                session.merge(db_record)
                stored_count += 1
            
            session.commit()
            
            logger.info("Assets stored successfully", count=stored_count)
            return stored_count
            
        except Exception as e:
            session.rollback()
            logger.error("Failed to store assets", error=str(e))
            raise
        finally:
            session.close()
    
    def export_to_csv(
        self,
        symbol: str,
        timeframe: str,
        output_path: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> str:
        """Export bar data to CSV file.
        
        Args:
            symbol: Symbol to export
            timeframe: Data timeframe
            output_path: Output file path
            start: Start datetime
            end: End datetime
            
        Returns:
            Path to exported file
        """
        bars = self.get_bars(symbol, timeframe, start, end)
        
        if not bars:
            logger.warning("No data to export", symbol=symbol, timeframe=timeframe)
            return ""
        
        # Convert to DataFrame
        data = []
        for bar in bars:
            data.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'trade_count': bar.trade_count,
                'vwap': bar.vwap
            })
        
        df = pd.DataFrame(data)
        
        # Generate output path if not provided
        if not output_path:
            filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = self.data_path / "exports" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        df.to_csv(output_path, index=False)
        
        logger.info(
            "Data exported to CSV",
            symbol=symbol,
            timeframe=timeframe,
            output_path=str(output_path),
            records=len(bars)
        )
        
        return str(output_path)
    
    def export_to_parquet(
        self,
        symbol: str,
        timeframe: str,
        output_path: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> str:
        """Export bar data to Parquet file.
        
        Args:
            symbol: Symbol to export
            timeframe: Data timeframe
            output_path: Output file path
            start: Start datetime
            end: End datetime
            
        Returns:
            Path to exported file
        """
        bars = self.get_bars(symbol, timeframe, start, end)
        
        if not bars:
            logger.warning("No data to export", symbol=symbol, timeframe=timeframe)
            return ""
        
        # Convert to DataFrame
        data = []
        for bar in bars:
            data.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'trade_count': bar.trade_count,
                'vwap': bar.vwap
            })
        
        df = pd.DataFrame(data)
        
        # Generate output path if not provided
        if not output_path:
            filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            output_path = self.data_path / "exports" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to Parquet
        df.to_parquet(output_path, index=False)
        
        logger.info(
            "Data exported to Parquet",
            symbol=symbol,
            timeframe=timeframe,
            output_path=str(output_path),
            records=len(bars)
        )
        
        return str(output_path)
    
    def get_available_symbols(self, timeframe: str) -> List[str]:
        """Get list of available symbols for a timeframe.
        
        Args:
            timeframe: Data timeframe
            
        Returns:
            List of available symbols
        """
        session = self.get_session()
        
        try:
            symbols = session.query(BarDataTable.symbol).filter(
                BarDataTable.timeframe == timeframe
            ).distinct().all()
            
            return [symbol[0] for symbol in symbols]
            
        except Exception as e:
            logger.error("Failed to get available symbols", error=str(e))
            raise
        finally:
            session.close()
    
    def get_data_range(
        self,
        symbol: str,
        timeframe: str
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get data range for a symbol and timeframe.
        
        Args:
            symbol: Symbol to check
            timeframe: Data timeframe
            
        Returns:
            Tuple of (start_date, end_date)
        """
        session = self.get_session()
        
        try:
            from sqlalchemy import func
            
            result = session.query(
                func.min(BarDataTable.timestamp),
                func.max(BarDataTable.timestamp)
            ).filter(
                BarDataTable.symbol == symbol,
                BarDataTable.timeframe == timeframe
            ).first()
            
            return result if result else (None, None)
            
        except Exception as e:
            logger.error(
                "Failed to get data range",
                symbol=symbol,
                timeframe=timeframe,
                error=str(e)
            )
            raise
        finally:
            session.close()
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """Clean up old data beyond retention period.
        
        Args:
            days_to_keep: Number of days to keep
            
        Returns:
            Number of records deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        session = self.get_session()
        
        try:
            # Delete old bar data
            deleted_bars = session.query(BarDataTable).filter(
                BarDataTable.timestamp < cutoff_date
            ).delete()
            
            # Delete old trade data
            deleted_trades = session.query(TradeDataTable).filter(
                TradeDataTable.timestamp < cutoff_date
            ).delete()
            
            # Delete old quote data
            deleted_quotes = session.query(QuoteDataTable).filter(
                QuoteDataTable.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            
            total_deleted = deleted_bars + deleted_trades + deleted_quotes
            
            logger.info(
                "Old data cleaned up",
                cutoff_date=cutoff_date,
                deleted_bars=deleted_bars,
                deleted_trades=deleted_trades,
                deleted_quotes=deleted_quotes,
                total_deleted=total_deleted
            )
            
            return total_deleted
            
        except Exception as e:
            session.rollback()
            logger.error("Failed to cleanup old data", error=str(e))
            raise
        finally:
            session.close()
    
    def close(self):
        """Close database connections."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
        logger.info("DataStorage closed")