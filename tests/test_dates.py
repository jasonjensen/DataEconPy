"""Date packing, and the agreement between DataEcon codes and TimeSeriesEconPy MITs.

Date conversion is done by the C library. What matters on the Python side is
that ``tsecon``'s ``MIT`` integers are the *same* integers the C library packs,
because the connector writes whole arrays of dates without converting them
element by element. If that ever stopped being true, series would come back
shifted in time, so it is checked here across every frequency.
"""

from __future__ import annotations

import ctypes
import datetime as dt

import pytest

from dataecon import _clib as C
from dataecon._consts import Frequency, freq_has_ppy, freq_ppy

from .conftest import requires_tsecon

ALL_FREQUENCIES = [
    Frequency.DAILY,
    Frequency.BDAILY,
    *range(Frequency.WEEKLY, Frequency.WEEKLY + 8),
    Frequency.MONTHLY,
    *range(Frequency.QUARTERLY, Frequency.QUARTERLY + 4),
    *range(Frequency.HALFYEARLY, Frequency.HALFYEARLY + 7),
    *range(Frequency.YEARLY, Frequency.YEARLY + 13),
]

YP_FREQUENCIES = [f for f in ALL_FREQUENCIES if freq_has_ppy(f)]
CALENDAR_FREQUENCIES = [f for f in ALL_FREQUENCIES if not freq_has_ppy(f)]


def pack_yp(freq: int, year: int, period: int) -> int:
    out = ctypes.c_int64()
    C.check(C.lib.de_pack_year_period_date(int(freq), year, period, ctypes.byref(out)))
    return out.value


def unpack_yp(freq: int, code: int) -> tuple[int, int]:
    year, period = ctypes.c_int32(), ctypes.c_uint32()
    C.check(
        C.lib.de_unpack_year_period_date(int(freq), code, ctypes.byref(year), ctypes.byref(period))
    )
    return year.value, period.value


def pack_cal(freq: int, year: int, month: int, day: int) -> int:
    out = ctypes.c_int64()
    C.check(C.lib.de_pack_calendar_date(int(freq), year, month, day, ctypes.byref(out)))
    return out.value


def unpack_cal(freq: int, code: int) -> tuple[int, int, int]:
    year, month, day = ctypes.c_int32(), ctypes.c_uint32(), ctypes.c_uint32()
    C.check(
        C.lib.de_unpack_calendar_date(
            int(freq), code, ctypes.byref(year), ctypes.byref(month), ctypes.byref(day)
        )
    )
    return year.value, month.value, day.value


class TestKnownCodes:
    """Anchor values that pin the epoch down."""

    def test_daily_epoch(self) -> None:
        # The epoch is chosen so that 0001-01-01 is day 1, matching Julia's
        # Dates.Date rata die and tsecon's MIT{Daily}.
        assert pack_cal(Frequency.DAILY, 1, 1, 1) == 1

    def test_daily_known_date(self) -> None:
        assert pack_cal(Frequency.DAILY, 2020, 1, 1) == 737425

    def test_quarterly_is_year_times_ppy_plus_period(self) -> None:
        assert pack_yp(Frequency.QUARTERLY_MAR, 2020, 1) == 2020 * 4
        assert pack_yp(Frequency.QUARTERLY_MAR, 2020, 3) == 2020 * 4 + 2

    def test_monthly_is_year_times_twelve(self) -> None:
        assert pack_yp(Frequency.MONTHLY, 2020, 1) == 2020 * 12

    def test_leap_day_round_trips(self) -> None:
        code = pack_cal(Frequency.DAILY, 2024, 2, 29)
        assert unpack_cal(Frequency.DAILY, code) == (2024, 2, 29)


class TestYearPeriodRoundTrip:
    @pytest.mark.parametrize("freq", YP_FREQUENCIES)
    @pytest.mark.parametrize("year", [-100, 0, 1, 1900, 1970, 2024, 3000])
    def test_pack_unpack_is_identity(self, freq: int, year: int) -> None:
        for period in range(1, freq_ppy(freq) + 1):
            code = pack_yp(freq, year, period)
            assert unpack_yp(freq, code) == (year, period)

    @pytest.mark.parametrize("freq", YP_FREQUENCIES)
    def test_consecutive_periods_are_consecutive_codes(self, freq: int) -> None:
        ppy = freq_ppy(freq)
        codes = [pack_yp(freq, 2020, p) for p in range(1, ppy + 1)]
        assert codes == list(range(codes[0], codes[0] + ppy))

    @pytest.mark.parametrize("freq", YP_FREQUENCIES)
    def test_year_rolls_over_by_ppy(self, freq: int) -> None:
        ppy = freq_ppy(freq)
        assert pack_yp(freq, 2021, 1) - pack_yp(freq, 2020, 1) == ppy


class TestCalendarRoundTrip:
    @pytest.mark.parametrize("freq", CALENDAR_FREQUENCIES)
    def test_pack_unpack_is_stable(self, freq: int) -> None:
        """Packing an unpacked date gives the code back."""
        for code in range(730000, 730000 + 40):
            year, month, day = unpack_cal(freq, code)
            assert pack_cal(freq, year, month, day) == code

    def test_daily_advances_one_per_day(self) -> None:
        first = pack_cal(Frequency.DAILY, 2024, 3, 1)
        for offset in range(60):
            date = dt.date(2024, 3, 1) + dt.timedelta(days=offset)
            assert pack_cal(Frequency.DAILY, date.year, date.month, date.day) == first + offset

    def test_business_daily_skips_weekends(self) -> None:
        friday = pack_cal(Frequency.BDAILY, 2024, 6, 14)
        monday = pack_cal(Frequency.BDAILY, 2024, 6, 17)
        assert monday - friday == 1

    def test_business_daily_rejects_a_weekend_date(self) -> None:
        from dataecon.errors import DEInexactError

        with pytest.raises(DEInexactError):
            pack_cal(Frequency.BDAILY, 2024, 6, 15)  # a Saturday

    @pytest.mark.parametrize("end_day", range(1, 8))
    def test_weekly_week_ends_on_the_requested_weekday(self, end_day: int) -> None:
        """The last day of each week must be the weekday named by the code."""
        freq = Frequency.WEEKLY + end_day
        start = dt.date(2024, 6, 10)
        codes = [
            pack_cal(
                freq,
                (start + dt.timedelta(days=k)).year,
                (start + dt.timedelta(days=k)).month,
                (start + dt.timedelta(days=k)).day,
            )
            for k in range(15)
        ]
        boundaries = [k for k in range(1, 15) if codes[k] != codes[k - 1]]
        assert boundaries, "no week boundary in a fortnight"
        for k in boundaries:
            last_day_of_week = start + dt.timedelta(days=k - 1)
            assert last_day_of_week.isoweekday() == end_day

    @pytest.mark.parametrize("end_day", range(1, 8))
    def test_eight_consecutive_days_span_two_weeks(self, end_day: int) -> None:
        freq = Frequency.WEEKLY + end_day
        codes = {pack_cal(freq, 2024, 6, day) for day in range(10, 18)}
        assert len(codes) == 2


@requires_tsecon
class TestAgreementWithTimeSeriesEconPy:
    """``MIT`` integers and DataEcon date codes must be the same integers."""

    @pytest.mark.parametrize("freq", ALL_FREQUENCIES)
    def test_encoding_verification_passes(self, freq: int) -> None:
        from dataecon.interop._tsecon import verify_frequency_encoding

        verify_frequency_encoding.cache_clear()
        verify_frequency_encoding(int(freq))

    @pytest.mark.parametrize("freq", YP_FREQUENCIES)
    @pytest.mark.parametrize("year", [1900, 1970, 2024])
    def test_year_period_codes_match(self, freq: int, year: int) -> None:
        import tsecon as ts

        from dataecon.interop._tsecon import freq_from_code

        frequency = freq_from_code(freq)
        for period in range(1, freq_ppy(freq) + 1):
            assert int(ts.MIT.from_yp(frequency, year, period)) == pack_yp(freq, year, period)

    @pytest.mark.parametrize(
        "date", [dt.date(1970, 1, 5), dt.date(2000, 3, 1), dt.date(2024, 6, 12)]
    )
    def test_daily_codes_match(self, date: dt.date) -> None:
        import tsecon as ts

        assert int(ts.daily(date)) == pack_cal(Frequency.DAILY, date.year, date.month, date.day)

    @pytest.mark.parametrize("date", [dt.date(1970, 1, 5), dt.date(2024, 6, 12)])
    def test_business_daily_codes_match(self, date: dt.date) -> None:
        import tsecon as ts

        assert int(ts.bdaily(date)) == pack_cal(Frequency.BDAILY, date.year, date.month, date.day)

    @pytest.mark.parametrize("end_day", range(1, 8))
    def test_weekly_codes_match(self, end_day: int) -> None:
        import tsecon as ts

        date = dt.date(2024, 6, 12)
        freq = Frequency.WEEKLY + end_day
        assert int(ts.weekly(date, end_day)) == pack_cal(freq, date.year, date.month, date.day)


@requires_tsecon
class TestFrequencyCodeMapping:
    """The frequency code mapping must survive a round trip in both directions."""

    @pytest.mark.parametrize("freq", ALL_FREQUENCIES)
    def test_code_to_frequency_and_back(self, freq: int) -> None:
        from dataecon.interop._tsecon import freq_from_code, freq_to_code

        assert freq_to_code(freq_from_code(freq)) == _canonical(int(freq))

    def test_family_defaults_map_to_the_family_default_frequency(self) -> None:
        import tsecon as ts

        from dataecon.interop._tsecon import freq_from_code

        # A bare family code carries no end period; the C library treats it the
        # same as the family's default, so the mapping does too.
        assert freq_from_code(Frequency.WEEKLY) == ts.Weekly(7)
        assert freq_from_code(Frequency.QUARTERLY) == ts.Quarterly(3)
        assert freq_from_code(Frequency.HALFYEARLY) == ts.HalfYearly(6)
        assert freq_from_code(Frequency.YEARLY) == ts.Yearly(12)


def _canonical(freq: int) -> int:
    """Map a bare family code onto the equivalent explicit end-period code."""
    for family, modulus in (
        (Frequency.WEEKLY, 7),
        (Frequency.QUARTERLY, 3),
        (Frequency.HALFYEARLY, 6),
        (Frequency.YEARLY, 12),
    ):
        if freq & family and freq - family == 0:
            return int(family) + modulus
    if freq == Frequency.MONTHLY + 1:
        return int(Frequency.MONTHLY)
    return freq
