# Write a reference .daec database using the Julia connector.
#
# Run as:   julia --project=. tests/cross/julia_write.jl <output.daec>
#
# The Python test suite reads the resulting file and checks every value, which
# is the strongest available statement that DataEconPy reads what
# TimeSeriesEcon.jl writes. The contents mirror `expected_julia_contents()` in
# tests/test_julia.py -- keep the two in step.

using TimeSeriesEcon
const DE = TimeSeriesEcon.DataEcon

function build()
    Workspace(
        # --- scalars -------------------------------------------------------
        i64   = Int64(-1234567890123),
        i32   = Int32(-2000000000),
        u16   = UInt16(65000),
        f64   = 3.141592653589793,
        f32   = Float32(2.5),
        c128  = 1.5 + 2.25im,
        text  = "hello .daec",
        # A Bool is stored as a 1-byte signed integer with no attribute, because
        # the connector only records `jtype` when the stored value's type differs
        # from the original -- and `_to_de_scalar_val(::Integer)` is the identity.
        # It therefore reads back on the Python side as an integer, not a bool.
        flag  = true,
        # A Symbol *is* converted on the way out (to a String), so this one does
        # carry `jtype`, which is the attribute path worth exercising.
        sym   = :baseline,

        # --- dates ---------------------------------------------------------
        dates = Workspace(
            q2020Q1   = 2020Q1,
            m1999M12  = 1999M12,
            y2024     = 2024Y,
            d20240229 = daily("2024-02-29"),
            b20240612 = bdaily("2024-06-12"),
            dur       = 2021Q1 - 2020Q1,
        ),

        # --- 1-d -----------------------------------------------------------
        vectors = Workspace(
            vec_f64   = [1.5, -2.5, 0.0, 1e10, -3.25],
            vec_i32   = Int32[1, -2, 3, -4],
            vec_str   = ["alpha", "beta", "gamma"],
            vec_empty = Float64[],
            rng_plain = 1:6,
            rng_dates = 2020Q1:2021Q4,
        ),

        # --- time series ---------------------------------------------------
        series = Workspace(
            ts_q   = TSeries(2020Q1, collect(0.0:7.0)),
            ts_d   = TSeries(daily("2024-01-15"), [10.5, 11.5, 12.5]),
            ts_m   = TSeries(2023M11, [-1.0, -2.0, -3.0, -4.0]),
            ts_i32 = TSeries(2020Q1, Int32[5, 6, 7, 8]),
        ),

        # --- 2-d and N-d ---------------------------------------------------
        arrays = Workspace(
            mat  = reshape(collect(1.0:6.0), 2, 3),
            mvts = MVTSeries(2021Q1, [:a, :b, :c], reshape(collect(1.0:12.0), 4, 3)),
            nd   = reshape(collect(0.0:23.0), 2, 3, 4),
        ),

        # --- nesting -------------------------------------------------------
        nested = Workspace(inner = Workspace(leaf = 42.0)),
    )
end

function main()
    length(ARGS) == 1 || error("usage: julia julia_write.jl <output.daec>")
    path = ARGS[1]
    isfile(path) && rm(path)

    DE.opendaec(path; write=true, truncate=true) do de
        DE.writedb(de, build())
        # An attribute, for the attribute round-trip check.
        DE.set_attribute(de, "/series/ts_q", "units", "billions")
    end

    println("wrote $path with TimeSeriesEcon ", pkgversion(TimeSeriesEcon),
            " and libdaec ", unsafe_string(DE.C.de_version()))
end

main()
