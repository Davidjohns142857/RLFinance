"""
Module: data.feature_engineer
Description: Feature engineering for trading with technical indicators
Author: Claude Code
Date: 2025-11-18
Version: 1.0.0

References:
    - TA-Lib: Technical Analysis Library
    - Paper: "Advances in Financial Machine Learning" (Lopez de Prado, 2018)

This module implements technical indicators commonly used in trading:
- Trend indicators: SMA, EMA, MACD
- Momentum indicators: RSI, CCI, Williams %R
- Volatility indicators: Bollinger Bands, ATR
- Volume indicators: OBV, VWAP

Example:
    >>> from rl_trading_system.data.feature_engineer import FeatureEngineer
    >>> engineer = FeatureEngineer({'indicators': ['SMA', 'RSI', 'MACD']})
    >>> features = engineer.process(ohlcv_data)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

from rl_trading_system.core.base import BaseModule
from rl_trading_system.core.exceptions import FeatureEngineeringException
from rl_trading_system.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer(BaseModule):
    """
    Feature engineering for trading data

    Calculates technical indicators from OHLCV data.

    Attributes:
        indicators: List of indicator names to calculate
        params: Parameters for each indicator

    Example:
        >>> config = {
        ...     'indicators': ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS'],
        ...     'params': {
        ...         'SMA': {'period': 20},
        ...         'RSI': {'period': 14}
        ...     }
        ... }
        >>> engineer = FeatureEngineer(config)
        >>> df_with_features = engineer.process(df)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineer

        Args:
            config: Configuration dictionary
                - indicators (List[str]): Indicator names
                - params (Dict): Parameters for indicators
                - fillna (bool): Fill NaN values
        """
        # Set attributes before calling super().__init__ because _validate_config needs them
        self.indicators = config.get('indicators', [])
        self.params = config.get('params', {})
        self.fillna = config.get('fillna', True)

        super().__init__(config)

        logger.info(f"FeatureEngineer initialized with {len(self.indicators)} indicators")

    def _validate_config(self) -> None:
        """Validate configuration"""
        if not self.indicators:
            logger.warning("No indicators specified")

    def _initialize(self) -> None:
        """Initialize feature engineer"""
        # Set default parameters for each indicator
        self._set_default_params()

    def _set_default_params(self) -> None:
        """Set default parameters for indicators"""
        defaults = {
            'SMA': {'period': 20},
            'EMA': {'period': 12},
            'RSI': {'period': 14},
            'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
            'BBANDS': {'period': 20, 'std': 2},
            'ATR': {'period': 14},
            'CCI': {'period': 20},
            'WILLR': {'period': 14},
            'OBV': {},
            'VWAP': {}
        }

        for indicator in self.indicators:
            if indicator not in self.params:
                self.params[indicator] = defaults.get(indicator, {})

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with added indicator columns

        Raises:
            FeatureEngineeringException: If calculation fails
        """
        try:
            df = data.copy()

            # Validate required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise FeatureEngineeringException(
                    f"Missing required columns: {missing}",
                    context={'columns': df.columns.tolist()}
                )

            # Calculate each indicator
            for indicator in self.indicators:
                logger.debug(f"Calculating {indicator}")

                if indicator == 'SMA':
                    df = self._calculate_sma(df)
                elif indicator == 'EMA':
                    df = self._calculate_ema(df)
                elif indicator == 'RSI':
                    df = self._calculate_rsi(df)
                elif indicator == 'MACD':
                    df = self._calculate_macd(df)
                elif indicator == 'BBANDS':
                    df = self._calculate_bollinger_bands(df)
                elif indicator == 'ATR':
                    df = self._calculate_atr(df)
                elif indicator == 'CCI':
                    df = self._calculate_cci(df)
                elif indicator == 'WILLR':
                    df = self._calculate_williams_r(df)
                elif indicator == 'OBV':
                    df = self._calculate_obv(df)
                elif indicator == 'VWAP':
                    df = self._calculate_vwap(df)
                else:
                    logger.warning(f"Unknown indicator: {indicator}")

            # Fill NaN values
            if self.fillna:
                df = df.ffill().fillna(0)

            logger.info(f"Features calculated: {df.shape[1]} columns")
            return df

        except Exception as e:
            raise FeatureEngineeringException(
                f"Feature calculation failed: {e}",
                context={'indicators': self.indicators}
            )

    def _calculate_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Simple Moving Average

        Formula: SMA = sum(prices) / period

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with SMA column
        """
        period = self.params['SMA']['period']
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df

    def _calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average

        Formula: EMA = price * k + EMA(previous) * (1 - k)
        where k = 2 / (period + 1)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with EMA column
        """
        period = self.params['EMA']['period']
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Relative Strength Index

        Formula:
            RSI = 100 - (100 / (1 + RS))
            RS = Average Gain / Average Loss

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with RSI column
        """
        period = self.params['RSI']['period']

        # Calculate price changes
        delta = df['close'].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / (avg_loss + 1e-8)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Formula:
            MACD = EMA(fast) - EMA(slow)
            Signal = EMA(MACD, signal_period)
            Histogram = MACD - Signal

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with MACD, Signal, and Histogram columns
        """
        fast = self.params['MACD']['fast']
        slow = self.params['MACD']['slow']
        signal_period = self.params['MACD']['signal']

        # Calculate MACD line
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow

        # Calculate signal line
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']

        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands

        Formula:
            Middle Band = SMA(period)
            Upper Band = Middle Band + (std * standard_deviation)
            Lower Band = Middle Band - (std * standard_deviation)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with Bollinger Bands columns
        """
        period = self.params['BBANDS']['period']
        std_multiplier = self.params['BBANDS']['std']

        # Calculate middle band (SMA)
        df['bb_middle'] = df['close'].rolling(window=period).mean()

        # Calculate standard deviation
        rolling_std = df['close'].rolling(window=period).std()

        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (std_multiplier * rolling_std)
        df['bb_lower'] = df['bb_middle'] - (std_multiplier * rolling_std)

        # Calculate bandwidth
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        return df

    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Average True Range

        Formula:
            TR = max(high - low, abs(high - close_prev), abs(low - close_prev))
            ATR = EMA(TR, period)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with ATR column
        """
        period = self.params['ATR']['period']

        # Calculate true range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate ATR
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()

        return df

    def _calculate_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Commodity Channel Index

        Formula:
            CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
            Typical Price = (High + Low + Close) / 3

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with CCI column
        """
        period = self.params['CCI']['period']

        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Calculate SMA of typical price
        sma_tp = typical_price.rolling(window=period).mean()

        # Calculate mean deviation
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )

        # Calculate CCI
        df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)

        return df

    def _calculate_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Williams %R

        Formula:
            %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with Williams %R column
        """
        period = self.params['WILLR']['period']

        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()

        df[f'willr_{period}'] = (
            (highest_high - df['close']) / (highest_high - lowest_low + 1e-8) * -100
        )

        return df

    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On-Balance Volume

        Formula:
            If close > close_prev: OBV = OBV_prev + volume
            If close < close_prev: OBV = OBV_prev - volume
            If close == close_prev: OBV = OBV_prev

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with OBV column
        """
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        df['obv'] = obv
        return df

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Weighted Average Price

        Formula:
            VWAP = sum(typical_price * volume) / sum(volume)
            Typical Price = (High + Low + Close) / 3

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with VWAP column
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        return df

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names that will be created

        Returns:
            List of feature column names
        """
        features = []

        for indicator in self.indicators:
            if indicator == 'SMA':
                period = self.params['SMA']['period']
                features.append(f'sma_{period}')
            elif indicator == 'EMA':
                period = self.params['EMA']['period']
                features.append(f'ema_{period}')
            elif indicator == 'RSI':
                period = self.params['RSI']['period']
                features.append(f'rsi_{period}')
            elif indicator == 'MACD':
                features.extend(['macd', 'macd_signal', 'macd_hist'])
            elif indicator == 'BBANDS':
                features.extend(['bb_middle', 'bb_upper', 'bb_lower', 'bb_width'])
            elif indicator == 'ATR':
                period = self.params['ATR']['period']
                features.append(f'atr_{period}')
            elif indicator == 'CCI':
                period = self.params['CCI']['period']
                features.append(f'cci_{period}')
            elif indicator == 'WILLR':
                period = self.params['WILLR']['period']
                features.append(f'willr_{period}')
            elif indicator == 'OBV':
                features.append('obv')
            elif indicator == 'VWAP':
                features.append('vwap')

        return features
