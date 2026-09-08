/*
 * Reference driver for cross-checking DataEconPy against the DataEcon C library.
 *
 * This program links directly against libdaec and writes/reads .daec files using
 * exactly the conventions the Julia connector uses (TimeSeriesEcon.jl's
 * src/dataecon/I.jl): the same choice of axis kind per object type, the same
 * `jtype`/`jeltype` attributes, and column-major element order.
 *
 * Two commands:
 *
 *   c_reference write <file>   write a reference database covering every
 *                              storage class, element type and frequency
 *   c_reference dump  <file>   print a canonical textual dump of a database
 *
 * The Python test suite runs both directions: it reads the file this program
 * writes, and dumps a file Python wrote to compare against this program's dump.
 * Because the dump format is shared, any disagreement about layout, byte order
 * or axis interpretation shows up as a diff.
 */

#include "daec.h"

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *what, int rc)
{
    char msg[512];
    de_error(msg, sizeof msg);
    fprintf(stderr, "FATAL: %s failed (rc=%d): %s\n", what, rc, msg);
    exit(EXIT_FAILURE);
}

#define RUN(call)                            \
    do {                                     \
        int _rc = (call);                    \
        if (_rc != DE_SUCCESS)               \
            fail(#call, _rc);                \
    } while (0)

/* ---------------------------------------------------------------- writing */

static obj_id_t catalog(de_file de, obj_id_t pid, const char *name)
{
    obj_id_t id;
    RUN(de_new_catalog(de, pid, name, &id));
    return id;
}

static void write_scalars(de_file de, obj_id_t pid)
{
    obj_id_t id;

    int64_t i64 = -1234567890123LL;
    RUN(de_store_scalar(de, pid, "i64", type_signed, freq_none, sizeof i64, &i64, &id));

    int32_t i32 = -2000000000;
    RUN(de_store_scalar(de, pid, "i32", type_signed, freq_none, sizeof i32, &i32, &id));

    int16_t i16 = -32000;
    RUN(de_store_scalar(de, pid, "i16", type_signed, freq_none, sizeof i16, &i16, &id));

    int8_t i8 = -128;
    RUN(de_store_scalar(de, pid, "i8", type_signed, freq_none, sizeof i8, &i8, &id));

    uint64_t u64 = 18000000000000000000ULL;
    RUN(de_store_scalar(de, pid, "u64", type_unsigned, freq_none, sizeof u64, &u64, &id));

    uint16_t u16 = 65000;
    RUN(de_store_scalar(de, pid, "u16", type_unsigned, freq_none, sizeof u16, &u16, &id));

    double f64 = 3.141592653589793;
    RUN(de_store_scalar(de, pid, "f64", type_float, freq_none, sizeof f64, &f64, &id));

    float f32 = 2.5f;
    RUN(de_store_scalar(de, pid, "f32", type_float, freq_none, sizeof f32, &f32, &id));

    double complex c128 = 1.5 + 2.25 * I;
    RUN(de_store_scalar(de, pid, "c128", type_complex, freq_none, sizeof c128, &c128, &id));

    float complex c64 = 0.5f + (-1.25f) * I;
    RUN(de_store_scalar(de, pid, "c64", type_complex, freq_none, sizeof c64, &c64, &id));

    char text[] = "hello .daec";
    RUN(de_store_scalar(de, pid, "text", type_string, freq_none,
                        sizeof text, text, &id));

    /* A Bool, exactly as I.jl stores it: one signed byte plus a jtype attribute. */
    int8_t flag = 1;
    RUN(de_store_scalar(de, pid, "flag", type_signed, freq_none, sizeof flag, &flag, &id));
    RUN(de_set_attribute(de, id, "jtype", "Bool"));
}

static void write_dates(de_file de, obj_id_t pid)
{
    obj_id_t id;
    date_t d;

    RUN(de_pack_year_period_date(freq_quarterly_dec, 2020, 1, &d));
    RUN(de_store_scalar(de, pid, "q2020Q1", type_date, freq_quarterly_dec, sizeof d, &d, &id));

    RUN(de_pack_year_period_date(freq_monthly, 1999, 12, &d));
    RUN(de_store_scalar(de, pid, "m1999M12", type_date, freq_monthly, sizeof d, &d, &id));

    RUN(de_pack_year_period_date(freq_yearly_dec, 2024, 1, &d));
    RUN(de_store_scalar(de, pid, "y2024", type_date, freq_yearly_dec, sizeof d, &d, &id));

    RUN(de_pack_year_period_date(freq_halfyearly_jun, 2021, 2, &d));
    RUN(de_store_scalar(de, pid, "h2021H2", type_date, freq_halfyearly_jun, sizeof d, &d, &id));

    RUN(de_pack_calendar_date(freq_daily, 2024, 2, 29, &d));
    RUN(de_store_scalar(de, pid, "d20240229", type_date, freq_daily, sizeof d, &d, &id));

    RUN(de_pack_calendar_date(freq_bdaily, 2024, 6, 12, &d));
    RUN(de_store_scalar(de, pid, "b20240612", type_date, freq_bdaily, sizeof d, &d, &id));

    RUN(de_pack_calendar_date(freq_weekly_sun, 2024, 6, 12, &d));
    RUN(de_store_scalar(de, pid, "w20240612", type_date, freq_weekly_sun, sizeof d, &d, &id));

    /* A Duration: a signed integer that carries a frequency. */
    int64_t dur = 12;
    RUN(de_store_scalar(de, pid, "dur", type_signed, freq_quarterly_dec, sizeof dur, &dur, &id));
}

static void write_vectors(de_file de, obj_id_t pid)
{
    obj_id_t id;
    axis_id_t ax;

    double vf[5] = {1.5, -2.5, 0.0, 1e10, -3.25};
    RUN(de_axis_plain(de, 5, &ax));
    RUN(de_store_tseries(de, pid, "vec_f64", type_vector, type_float, freq_none,
                         ax, sizeof vf, vf, &id));

    int32_t vi[4] = {1, -2, 3, -4};
    RUN(de_axis_plain(de, 4, &ax));
    RUN(de_store_tseries(de, pid, "vec_i32", type_vector, type_signed, freq_none,
                         ax, sizeof vi, vi, &id));

    /* A string vector, packed as I.jl packs it. */
    const char *words[3] = {"alpha", "beta", "gamma"};
    int64_t bufsize = -1;
    RUN(de_pack_strings(words, 3, NULL, &bufsize));
    char *packed = malloc((size_t)bufsize);
    RUN(de_pack_strings(words, 3, packed, &bufsize));
    RUN(de_axis_plain(de, 3, &ax));
    RUN(de_store_tseries(de, pid, "vec_str", type_vector, type_string, freq_none,
                         ax, bufsize, packed, &id));
    free(packed);

    /* An empty vector. */
    RUN(de_axis_plain(de, 0, &ax));
    RUN(de_store_tseries(de, pid, "vec_empty", type_vector, type_float, freq_none,
                         ax, 0, NULL, &id));

    /* A plain range: an axis and no element data at all. */
    RUN(de_axis_plain(de, 6, &ax));
    RUN(de_store_tseries(de, pid, "rng_plain", type_range, type_none, freq_none,
                         ax, 0, NULL, &id));

    /* A date range. */
    date_t first;
    RUN(de_pack_year_period_date(freq_quarterly_dec, 2020, 1, &first));
    RUN(de_axis_range(de, 8, freq_quarterly_dec, first, &ax));
    RUN(de_store_tseries(de, pid, "rng_dates", type_range, type_none, freq_none,
                         ax, 0, NULL, &id));
}

static void write_tseries(de_file de, obj_id_t pid)
{
    obj_id_t id;
    axis_id_t ax;
    date_t first;

    double q[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    RUN(de_pack_year_period_date(freq_quarterly_dec, 2020, 1, &first));
    RUN(de_axis_range(de, 8, freq_quarterly_dec, first, &ax));
    RUN(de_store_tseries(de, pid, "ts_q", type_tseries, type_float, freq_none,
                         ax, sizeof q, q, &id));

    double d[3] = {10.5, 11.5, 12.5};
    RUN(de_pack_calendar_date(freq_daily, 2024, 1, 15, &first));
    RUN(de_axis_range(de, 3, freq_daily, first, &ax));
    RUN(de_store_tseries(de, pid, "ts_d", type_tseries, type_float, freq_none,
                         ax, sizeof d, d, &id));

    double m[4] = {-1.0, -2.0, -3.0, -4.0};
    RUN(de_pack_year_period_date(freq_monthly, 2023, 11, &first));
    RUN(de_axis_range(de, 4, freq_monthly, first, &ax));
    RUN(de_store_tseries(de, pid, "ts_m", type_tseries, type_float, freq_none,
                         ax, sizeof m, m, &id));

    /* An integer-valued time series, which needs a jeltype attribute. */
    int32_t qi[4] = {5, 6, 7, 8};
    RUN(de_pack_year_period_date(freq_quarterly_dec, 2020, 1, &first));
    RUN(de_axis_range(de, 4, freq_quarterly_dec, first, &ax));
    RUN(de_store_tseries(de, pid, "ts_i32", type_tseries, type_signed, freq_none,
                         ax, sizeof qi, qi, &id));
}

static void write_matrices(de_file de, obj_id_t pid)
{
    obj_id_t id;
    axis_id_t ax1, ax2;

    /* Column-major 2x3: element (i,j) lives at i + j*2. */
    double mat[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    RUN(de_axis_plain(de, 2, &ax1));
    RUN(de_axis_plain(de, 3, &ax2));
    RUN(de_store_mvtseries(de, pid, "mat", type_matrix, type_float, freq_none,
                           ax1, ax2, sizeof mat, mat, &id));

    /* An mvtseries: date range down the rows, names across the columns. */
    double mv[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    date_t first;
    RUN(de_pack_year_period_date(freq_quarterly_dec, 2021, 1, &first));
    RUN(de_axis_range(de, 4, freq_quarterly_dec, first, &ax1));
    RUN(de_axis_names(de, 3, "a\nb\nc", &ax2));
    RUN(de_store_mvtseries(de, pid, "mvts", type_mvtseries, type_float, freq_none,
                           ax1, ax2, sizeof mv, mv, &id));
}

static void write_tensors(de_file de, obj_id_t pid)
{
    obj_id_t id;
    axis_id_t axes[3];
    double nd[24];
    for (int i = 0; i < 24; ++i)
        nd[i] = (double)i;

    RUN(de_axis_plain(de, 2, &axes[0]));
    RUN(de_axis_plain(de, 3, &axes[1]));
    RUN(de_axis_plain(de, 4, &axes[2]));
    RUN(de_store_ndtseries(de, pid, "nd", type_tensor, type_float, freq_none,
                           3, axes, sizeof nd, nd, &id));
}

static int do_write(const char *path)
{
    remove(path);
    de_file de;
    RUN(de_open(path, &de));

    write_scalars(de, 0);

    obj_id_t dates = catalog(de, 0, "dates");
    write_dates(de, dates);

    obj_id_t vectors = catalog(de, 0, "vectors");
    write_vectors(de, vectors);

    obj_id_t series = catalog(de, 0, "series");
    write_tseries(de, series);

    obj_id_t arrays = catalog(de, 0, "arrays");
    write_matrices(de, arrays);
    write_tensors(de, arrays);

    obj_id_t nested = catalog(de, 0, "nested");
    obj_id_t inner = catalog(de, nested, "inner");
    obj_id_t id;
    double leaf = 42.0;
    RUN(de_store_scalar(de, inner, "leaf", type_float, freq_none, sizeof leaf, &leaf, &id));

    /* Attributes on an object, for the attribute round-trip test. */
    obj_id_t ts_q;
    RUN(de_find_fullpath(de, "/series/ts_q", &ts_q));
    RUN(de_set_attribute(de, ts_q, "units", "billions"));
    RUN(de_set_attribute(de, ts_q, "source", "reference C driver"));

    RUN(de_close(de));
    return EXIT_SUCCESS;
}

/* ---------------------------------------------------------------- dumping */

static void print_scalar_value(type_t type, frequency_t freq, int64_t nbytes, const void *v)
{
    if (v == NULL || nbytes == 0) {
        printf("null");
        return;
    }
    switch (type) {
    case type_signed:
        switch (nbytes) {
        case 1: printf("%d", (int)*(const int8_t *)v); return;
        case 2: printf("%d", (int)*(const int16_t *)v); return;
        case 4: printf("%d", *(const int32_t *)v); return;
        case 8: printf("%lld", (long long)*(const int64_t *)v); return;
        default: break;
        }
        break;
    case type_unsigned:
        switch (nbytes) {
        case 1: printf("%u", (unsigned)*(const uint8_t *)v); return;
        case 2: printf("%u", (unsigned)*(const uint16_t *)v); return;
        case 4: printf("%u", *(const uint32_t *)v); return;
        case 8: printf("%llu", (unsigned long long)*(const uint64_t *)v); return;
        default: break;
        }
        break;
    case type_date:
        printf("%lld@%d", (long long)*(const int64_t *)v, (int)freq);
        return;
    case type_float:
        if (nbytes == 4) { printf("%.9g", (double)*(const float *)v); return; }
        if (nbytes == 8) { printf("%.17g", *(const double *)v); return; }
        break;
    case type_complex:
        if (nbytes == 8) {
            const float *p = v;
            printf("(%.9g,%.9g)", (double)p[0], (double)p[1]);
            return;
        }
        if (nbytes == 16) {
            const double *p = v;
            printf("(%.17g,%.17g)", p[0], p[1]);
            return;
        }
        break;
    case type_string:
        printf("\"%s\"", (const char *)v);
        return;
    default:
        break;
    }
    printf("<%d bytes>", (int)nbytes);
}

static void print_elements(type_t eltype, frequency_t elfreq,
                           int64_t count, int64_t nbytes, const void *v)
{
    if (count == 0 || v == NULL) {
        printf("[]");
        return;
    }
    printf("[");
    if (eltype == type_string) {
        const char **strvec = malloc((size_t)count * sizeof(char *));
        if (de_unpack_strings(v, nbytes, strvec, count) != DE_SUCCESS)
            fail("de_unpack_strings", -1);
        for (int64_t i = 0; i < count; ++i)
            printf("%s\"%s\"", i ? " " : "", strvec[i]);
        free(strvec);
    } else {
        int64_t width = nbytes / count;
        const char *p = v;
        for (int64_t i = 0; i < count; ++i) {
            if (i) printf(" ");
            print_scalar_value(eltype, elfreq, width, p + i * width);
        }
    }
    printf("]");
}

static void print_axis(const axis_t *ax)
{
    switch (ax->ax_type) {
    case axis_plain:
        printf("plain:%lld", (long long)ax->length);
        break;
    case axis_range:
        printf("range:%lld:%d:%lld", (long long)ax->length,
               (int)ax->frequency, (long long)ax->first);
        break;
    case axis_names: {
        /* Names are stored newline-separated; print them comma-separated so
           that one object stays on one line of the dump. */
        printf("names:%lld:", (long long)ax->length);
        for (const char *p = ax->names ? ax->names : ""; *p; ++p)
            putchar(*p == '\n' ? ',' : *p);
        break;
    }
    default:
        printf("?");
        break;
    }
}

#define ATTR_DELIM "\x1f"

/* strdup is POSIX, not C99; the build uses -std=c99, so provide our own. */
static char *dup_string(const char *text)
{
    size_t len = strlen(text) + 1;
    char *out = malloc(len);
    memcpy(out, text, len);
    return out;
}

static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* Split `text` on ATTR_DELIM in place, filling `out` with up to `max` pieces. */
static int split_delim(char *text, char **out, int max)
{
    int n = 0;
    char *p = text;
    while (n < max) {
        out[n++] = p;
        char *sep = strstr(p, ATTR_DELIM);
        if (sep == NULL)
            break;
        *sep = '\0';
        p = sep + strlen(ATTR_DELIM);
    }
    return n;
}

static void print_attributes(de_file de, obj_id_t id)
{
    int64_t n = 0;
    const char *names = NULL, *values = NULL;
    if (de_get_all_attributes(de, id, ATTR_DELIM, &n, &names, &values) != DE_SUCCESS) {
        de_clear_error();
        return;
    }
    if (n == 0 || names == NULL || values == NULL)
        return;

    /* The library hands back all names and all values as two joined strings,
       in unspecified order. Pair them up and sort, so the dump is stable. */
    char *nbuf = dup_string(names);
    char *vbuf = dup_string(values);
    char *nparts[64], *vparts[64];
    int nn = split_delim(nbuf, nparts, 64);
    int nv = split_delim(vbuf, vparts, 64);

    char *pairs[64];
    int np = nn < nv ? nn : nv;
    for (int i = 0; i < np; ++i) {
        size_t len = strlen(nparts[i]) + strlen(vparts[i]) + 2;
        pairs[i] = malloc(len);
        snprintf(pairs[i], len, "%s=%s", nparts[i], vparts[i]);
    }
    qsort(pairs, (size_t)np, sizeof pairs[0], cmp_str);

    printf(" attrs={");
    for (int i = 0; i < np; ++i) {
        printf("%s%s", i ? "," : "", pairs[i]);
        free(pairs[i]);
    }
    printf("}");

    free(nbuf);
    free(vbuf);
}

static void dump_object(de_file de, obj_id_t id);

static void dump_catalog(de_file de, obj_id_t pid)
{
    /* Collect ids first: the search statement must not be live while we issue
       other library calls against the same connection. */
    de_search search;
    RUN(de_list_catalog(de, pid, &search));
    obj_id_t ids[512];
    int n = 0;
    object_t obj;
    while (de_next_object(search, &obj) == DE_SUCCESS && n < 512)
        ids[n++] = obj.id;
    de_finalize_search(search);
    de_clear_error();

    for (int i = 0; i < n; ++i)
        dump_object(de, ids[i]);
}

static void dump_object(de_file de, obj_id_t id)
{
    object_t obj;
    RUN(de_load_object(de, id, &obj));
    class_t cls = obj.obj_class;

    const char *fullpath = NULL;
    RUN(de_get_object_info(de, id, &fullpath, NULL, NULL));
    char path[1024];
    snprintf(path, sizeof path, "%s", fullpath);

    if (cls == class_catalog) {
        printf("CATALOG %s", path);
        print_attributes(de, id);
        printf("\n");
        dump_catalog(de, id);
        return;
    }

    if (cls == class_scalar) {
        scalar_t s;
        RUN(de_load_scalar(de, id, &s));
        printf("SCALAR %s type=%d freq=%d value=", path, (int)s.object.obj_type,
               (int)s.frequency);
        print_scalar_value(s.object.obj_type, s.frequency, s.nbytes, s.value);
        print_attributes(de, id);
        printf("\n");
        return;
    }

    if (cls == class_tseries) {
        tseries_t t;
        RUN(de_load_tseries(de, id, &t));
        printf("VECTOR %s type=%d eltype=%d elfreq=%d axis=", path,
               (int)t.object.obj_type, (int)t.eltype, (int)t.elfreq);
        print_axis(&t.axis);
        printf(" values=");
        if (t.object.obj_type == type_range)
            printf("[]");
        else
            print_elements(t.eltype, t.elfreq, t.axis.length, t.nbytes, t.value);
        print_attributes(de, id);
        printf("\n");
        return;
    }

    if (cls == class_mvtseries) {
        mvtseries_t m;
        RUN(de_load_mvtseries(de, id, &m));
        printf("MATRIX %s type=%d eltype=%d elfreq=%d axis1=", path,
               (int)m.object.obj_type, (int)m.eltype, (int)m.elfreq);
        print_axis(&m.axis1);
        printf(" axis2=");
        print_axis(&m.axis2);
        printf(" values=");
        print_elements(m.eltype, m.elfreq, m.axis1.length * m.axis2.length,
                       m.nbytes, m.value);
        print_attributes(de, id);
        printf("\n");
        return;
    }

    if (cls == class_ndtseries) {
        ndtseries_t a;
        RUN(de_load_ndtseries(de, id, &a));
        printf("TENSOR %s type=%d eltype=%d elfreq=%d naxes=%lld", path,
               (int)a.object.obj_type, (int)a.eltype, (int)a.elfreq,
               (long long)a.naxes);
        int64_t count = 1;
        for (int64_t i = 0; i < a.naxes; ++i) {
            printf(" axis%lld=", (long long)i);
            print_axis(&a.axis[i]);
            count *= a.axis[i].length;
        }
        printf(" values=");
        print_elements(a.eltype, a.elfreq, count, a.nbytes, a.value);
        print_attributes(de, id);
        printf("\n");
        return;
    }

    printf("UNKNOWN %s class=%d\n", path, (int)cls);
}

static int do_dump(const char *path)
{
    de_file de;
    RUN(de_open_readonly(path, &de));
    dump_catalog(de, 0);
    RUN(de_close(de));
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s {write|dump} <file.daec>\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], "write") == 0)
        return do_write(argv[2]);
    if (strcmp(argv[1], "dump") == 0)
        return do_dump(argv[2]);
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    return EXIT_FAILURE;
}
