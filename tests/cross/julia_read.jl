# Read a .daec database written by DataEconPy and verify it from Julia.
#
# Run as:   julia --project=. tests/cross/julia_read.jl <input.daec>
#
# Exits non-zero and prints every mismatch if anything fails, so the Python test
# can simply assert on the exit status. The expected contents mirror
# `write_julia_fixture()` in tests/test_julia.py -- keep the two in step.

using TimeSeriesEcon
const DE = TimeSeriesEcon.DataEcon

const FAILURES = String[]

function check(label, got, want)
    ok = try
        got == want
    catch
        false
    end
    ok || push!(FAILURES, "$label: got $(repr(got)), expected $(repr(want))")
    return ok
end

function check_values(label, got, want)
    check(label * ".values", collect(got), collect(want))
end

function main()
    length(ARGS) == 1 || error("usage: julia julia_read.jl <input.daec>")
    data = DE.readdb(ARGS[1])

    # --- scalars -----------------------------------------------------------
    check("i64", data.i64, Int64(-1234567890123))
    check("f64", data.f64, 3.141592653589793)
    check("c128", data.c128, 1.5 + 2.25im)
    check("text", data.text, "hello .daec")

    # --- dates -------------------------------------------------------------
    check("dates.q2020Q1", data.dates.q2020Q1, 2020Q1)
    check("dates.m1999M12", data.dates.m1999M12, 1999M12)
    check("dates.y2024", data.dates.y2024, 2024Y)
    check("dates.d20240229", data.dates.d20240229, daily("2024-02-29"))
    check("dates.b20240612", data.dates.b20240612, bdaily("2024-06-12"))

    # --- 1-d ---------------------------------------------------------------
    check_values("vectors.vec_f64", data.vectors.vec_f64, [1.5, -2.5, 0.0, 1e10, -3.25])
    check_values("vectors.vec_i32", data.vectors.vec_i32, Int32[1, -2, 3, -4])
    check_values("vectors.vec_str", data.vectors.vec_str, ["alpha", "beta", "gamma"])
    check("vectors.rng_dates", data.vectors.rng_dates, 2020Q1:2021Q4)

    # --- time series -------------------------------------------------------
    ts_q = data.series.ts_q
    check("series.ts_q.firstdate", firstdate(ts_q), 2020Q1)
    # N.B. `frequencyof` returns the concrete `Quarterly{3}`, and `Quarterly` on
    # its own is a UnionAll, so `==` between them is false. Compare with `<:`.
    check("series.ts_q.frequency", frequencyof(ts_q) <: Quarterly, true)
    check_values("series.ts_q", ts_q.values, collect(0.0:7.0))

    ts_d = data.series.ts_d
    check("series.ts_d.firstdate", firstdate(ts_d), daily("2024-01-15"))
    check_values("series.ts_d", ts_d.values, [10.5, 11.5, 12.5])

    ts_m = data.series.ts_m
    check("series.ts_m.firstdate", firstdate(ts_m), 2023M11)

    # --- 2-d and N-d -------------------------------------------------------
    # The orientation check matters most: a column-major/row-major mix-up shows
    # up here as a transposed matrix rather than as an error.
    check("arrays.mat", data.arrays.mat, reshape(collect(1.0:6.0), 2, 3))

    mvts = data.arrays.mvts
    check("arrays.mvts.firstdate", firstdate(mvts), 2021Q1)
    check("arrays.mvts.columns", collect(colnames(mvts)), [:a, :b, :c])
    check("arrays.mvts.values", mvts.values, reshape(collect(1.0:12.0), 4, 3))

    check("arrays.nd", data.arrays.nd, reshape(collect(0.0:23.0), 2, 3, 4))

    # --- nesting and attributes -------------------------------------------
    check("nested.inner.leaf", data.nested.inner.leaf, 42.0)

    DE.opendaec(ARGS[1]) do de
        check("attribute units", DE.get_attribute(de, "/series/ts_q", "units"), "billions")
    end

    if isempty(FAILURES)
        println("OK: all checks passed reading $(ARGS[1]) from Julia")
        exit(0)
    end
    println("FAILED: $(length(FAILURES)) check(s)")
    for failure in FAILURES
        println("  ", failure)
    end
    exit(1)
end

main()
