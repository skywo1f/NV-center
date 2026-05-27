#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>

#include "pico/stdlib.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include "hardware/i2c.h"

// ============================================================
// Pico <-> ADF4351 wiring
// ============================================================
#define PIN_SPI_SCK   18
#define PIN_SPI_MOSI  19
#define PIN_LE        17
#define PIN_CE        16

#define SPI_PORT      spi0
#define SPI_BAUD      1000000u

// ============================================================
// ADF4351 reference setup
// ============================================================
#define F_REF_HZ             25000000ULL
#define DEFAULT_R_COUNTER    1u
#define CHANNEL_SPACING_HZ   100000ULL

// ============================================================
// TSL2591 wiring / setup
// ============================================================
#define I2C_PORT       i2c0
#define I2C_SDA_PIN    0
#define I2C_SCL_PIN    1
#define I2C_BAUD_HZ    400000u

#define TSL2591_ADDR   0x29
#define TSL2591_CMD    0xA0

#define REG_ENABLE     0x00
#define REG_CONTROL    0x01
#define REG_ID         0x12
#define REG_STATUS     0x13
#define REG_CHAN0_LOW  0x14
#define REG_CHAN0_HIGH 0x15
#define REG_CHAN1_LOW  0x16
#define REG_CHAN1_HIGH 0x17

#define ENABLE_PON     (1u << 0)
#define ENABLE_AEN     (1u << 1)

#ifndef PICO_DEFAULT_LED_PIN
#define PICO_DEFAULT_LED_PIN 25
#endif

typedef enum {
    TSL2591_GAIN_LOW  = 0x00,
    TSL2591_GAIN_MED  = 0x10,
    TSL2591_GAIN_HIGH = 0x20,
    TSL2591_GAIN_MAX  = 0x30
} tsl2591_gain_t;

typedef enum {
    TSL2591_IT_100MS = 0x00,
    TSL2591_IT_200MS = 0x01,
    TSL2591_IT_300MS = 0x02,
    TSL2591_IT_400MS = 0x03,
    TSL2591_IT_500MS = 0x04,
    TSL2591_IT_600MS = 0x05
} tsl2591_it_t;

typedef enum {
    RUN_MODE_ODMR = 0,
    RUN_MODE_STREAM = 1
} run_mode_t;

// ============================================================
// Measurement settings
// ============================================================
#define SENSOR_GAIN_DEFAULT               TSL2591_GAIN_MED
#define SENSOR_IT_DEFAULT                 TSL2591_IT_200MS
#define SENSOR_EXTRA_WAIT_MS_DEFAULT      20u

#define USE_CH0_METRIC                    1
#define FREQ_SETTLE_MS_DEFAULT            100u
#define RF_ON_SETTLE_MS_DEFAULT           0u
#define RF_OFF_SETTLE_MS_DEFAULT          0u
#define MEAS_SETTLE_DISCARD_SAMPLES_DEFAULT   1u
#define MEAS_AVG_SAMPLES_PER_STATE_DEFAULT    4u

// ============================================================
// Globals
// ============================================================
static uint32_t g_adf_regs[6];
static bool g_adf_regs_valid = false;
static run_mode_t g_mode = RUN_MODE_ODMR;
static bool g_stream_enabled = false;
static tsl2591_gain_t g_sensor_gain = SENSOR_GAIN_DEFAULT;
static tsl2591_it_t g_sensor_it = SENSOR_IT_DEFAULT;
static uint32_t g_sensor_extra_wait_ms = SENSOR_EXTRA_WAIT_MS_DEFAULT;

static uint32_t g_freq_settle_ms = FREQ_SETTLE_MS_DEFAULT;
static uint32_t g_rf_on_settle_ms = RF_ON_SETTLE_MS_DEFAULT;
static uint32_t g_rf_off_settle_ms = RF_OFF_SETTLE_MS_DEFAULT;
static uint32_t g_meas_settle_discard_samples = MEAS_SETTLE_DISCARD_SAMPLES_DEFAULT;
static uint32_t g_meas_avg_samples_per_state = MEAS_AVG_SAMPLES_PER_STATE_DEFAULT;
static bool tsl2591_set_control(tsl2591_gain_t gain, tsl2591_it_t it);
static uint32_t tsl2591_integration_ms(tsl2591_it_t it);
// ============================================================
// Utility helpers
// ============================================================
static void str_to_lower_inplace(char *s) {
    while (*s) {
        *s = (char)tolower((unsigned char)*s);
        s++;
    }
}
static bool parse_u32_arg(const char *s, uint32_t *out) {
    char *endptr = NULL;
    unsigned long v;

    if (s == NULL || out == NULL) {
        return false;
    }

    while (*s != '\0' && isspace((unsigned char)*s)) {
        s++;
    }

    if (*s == '\0') {
        return false;
    }

    v = strtoul(s, &endptr, 10);
    if (endptr == s) {
        return false;
    }

    while (*endptr != '\0' && isspace((unsigned char)*endptr)) {
        endptr++;
    }

    if (*endptr != '\0') {
        return false;
    }

    *out = (uint32_t)v;
    return true;
}

static bool tsl2591_it_from_ms(uint32_t it_ms, tsl2591_it_t *it_out) {
    if (it_out == NULL) {
        return false;
    }

    switch (it_ms) {
        case 100: *it_out = TSL2591_IT_100MS; return true;
        case 200: *it_out = TSL2591_IT_200MS; return true;
        case 300: *it_out = TSL2591_IT_300MS; return true;
        case 400: *it_out = TSL2591_IT_400MS; return true;
        case 500: *it_out = TSL2591_IT_500MS; return true;
        case 600: *it_out = TSL2591_IT_600MS; return true;
        default: return false;
    }
}

static bool apply_sensor_control_runtime(void) {
    if (!tsl2591_set_control(g_sensor_gain, g_sensor_it)) {
        return false;
    }
    return true;
}

static void print_config(void) {
    printf(
        "CFGSTATE,it_ms=%lu,extra_wait_ms=%lu,freq_settle_ms=%lu,"
        "rf_on_settle_ms=%lu,rf_off_settle_ms=%lu,discard=%lu,avg=%lu\r\n",
        (unsigned long)tsl2591_integration_ms(g_sensor_it),
        (unsigned long)g_sensor_extra_wait_ms,
        (unsigned long)g_freq_settle_ms,
        (unsigned long)g_rf_on_settle_ms,
        (unsigned long)g_rf_off_settle_ms,
        (unsigned long)g_meas_settle_discard_samples,
        (unsigned long)g_meas_avg_samples_per_state
    );
}
static void trim_inplace(char *s) {
    size_t len;
    size_t start = 0;

    while (s[start] != '\0' && isspace((unsigned char)s[start])) {
        start++;
    }

    if (start > 0) {
        memmove(s, s + start, strlen(s + start) + 1);
    }

    len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[len - 1] = '\0';
        len--;
    }
}

static int parse_frequency_hz(const char *input, uint64_t *freq_hz_out) {
    char buf[64];
    char *endptr;
    double value;
    uint64_t multiplier = 1ULL;

    if (input == NULL || freq_hz_out == NULL) {
        return -1;
    }

    strncpy(buf, input, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    trim_inplace(buf);
    str_to_lower_inplace(buf);

    if (buf[0] == '\0') {
        return -2;
    }

    value = strtod(buf, &endptr);
    if (endptr == buf) {
        return -3;
    }

    while (*endptr != '\0' && isspace((unsigned char)*endptr)) {
        endptr++;
    }

    if (*endptr == '\0' || strcmp(endptr, "hz") == 0) {
        multiplier = 1ULL;
    } else if (strcmp(endptr, "khz") == 0) {
        multiplier = 1000ULL;
    } else if (strcmp(endptr, "mhz") == 0) {
        multiplier = 1000000ULL;
    } else if (strcmp(endptr, "ghz") == 0) {
        multiplier = 1000000000ULL;
    } else {
        return -4;
    }

    if (value <= 0.0) {
        return -5;
    }

    *freq_hz_out = (uint64_t)(value * (double)multiplier + 0.5);
    return 0;
}

static uint32_t gcd_u32(uint32_t a, uint32_t b) {
    if (a == 0u) return (b == 0u) ? 1u : b;
    if (b == 0u) return a;

    while (b != 0u) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

static void led_init(void) {
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    gpio_put(PICO_DEFAULT_LED_PIN, 0);
}

static void led_blink(int count, int on_ms, int off_ms) {
    for (int i = 0; i < count; i++) {
        gpio_put(PICO_DEFAULT_LED_PIN, 1);
        sleep_ms(on_ms);
        gpio_put(PICO_DEFAULT_LED_PIN, 0);
        sleep_ms(off_ms);
    }
}

// ============================================================
// ADF4351 helpers
// ============================================================
static inline void le_pulse(void) {
    gpio_put(PIN_LE, 1);
    sleep_us(1);
    gpio_put(PIN_LE, 0);
    sleep_us(1);
}

static void adf_write32(uint32_t w) {
    uint8_t b[4];
    b[0] = (uint8_t)((w >> 24) & 0xFF);
    b[1] = (uint8_t)((w >> 16) & 0xFF);
    b[2] = (uint8_t)((w >>  8) & 0xFF);
    b[3] = (uint8_t)((w >>  0) & 0xFF);
    spi_write_blocking(SPI_PORT, b, 4);
    le_pulse();
}

static void adf_program_regs(const uint32_t regs[6]) {
    adf_write32(regs[5]);
    adf_write32(regs[4]);
    adf_write32(regs[3]);
    adf_write32(regs[2]);
    adf_write32(regs[1]);
    adf_write32(regs[0]);
}

static int adf4351_build_regs(
    uint64_t fout_hz,
    uint64_t fref_hz,
    uint32_t r_counter,
    uint32_t regs[6]
) {
    if (regs == NULL) {
        return -1;
    }

    if (fout_hz < 35000000ULL || fout_hz > 4400000000ULL) {
        return -2;
    }

    if (fref_hz == 0ULL) {
        return -3;
    }

    r_counter &= 0x3FFu;
    if (r_counter == 0u) {
        r_counter = 1u;
    }

    const uint64_t f_pfd = fref_hz / (uint64_t)r_counter;
    if (f_pfd == 0ULL) {
        return -4;
    }

    uint32_t rf_div_sel = 0u;
    uint64_t vco_hz = fout_hz;

    while ((vco_hz < 2200000000ULL) && (rf_div_sel < 6u)) {
        vco_hz <<= 1;
        rf_div_sel++;
    }

    if (vco_hz < 2200000000ULL || vco_hz > 4400000000ULL) {
        return -5;
    }

    uint32_t int_val = (uint32_t)(vco_hz / f_pfd);
    uint64_t rem = vco_hz - ((uint64_t)int_val * f_pfd);

    uint32_t prescaler_8_9 = (int_val >= 75u) ? 1u : 0u;
    if ((!prescaler_8_9 && int_val < 23u) || (prescaler_8_9 && int_val < 75u)) {
        return -6;
    }

    uint32_t mod = (uint32_t)(f_pfd / CHANNEL_SPACING_HZ);
    if (mod < 2u) {
        mod = 2u;
    }
    if (mod > 4095u) {
        return -7;
    }

    uint32_t frac = (uint32_t)((rem * (uint64_t)mod + (f_pfd / 2ULL)) / f_pfd);

    if (frac >= mod) {
        frac = 0u;
        int_val += 1u;
    }

    uint32_t ldf;
    uint32_t ldp;

    if (frac == 0u) {
        mod = 2u;
        ldf = 1u;
        ldp = 1u;
    } else {
        uint32_t g = gcd_u32(frac, mod);
        frac /= g;
        mod /= g;
        ldf = 0u;
        ldp = 0u;
    }

    uint32_t band_sel_clkdiv = (uint32_t)((f_pfd + 124999ULL) / 125000ULL);
    if (band_sel_clkdiv < 1u) {
        band_sel_clkdiv = 1u;
    }
    if (band_sel_clkdiv > 255u) {
        band_sel_clkdiv = 255u;
    }

    regs[0] = 0u;
    regs[0] |= ((int_val & 0xFFFFu) << 15);
    regs[0] |= ((frac    & 0x0FFFu) << 3);
    regs[0] |= 0u;

    regs[1] = 0u;
    regs[1] |= (0u << 28);
    regs[1] |= ((prescaler_8_9 & 0x1u) << 27);
    regs[1] |= ((1u & 0x0FFFu) << 15);
    regs[1] |= ((mod & 0x0FFFu) << 3);
    regs[1] |= 1u;

    regs[2] = 0u;
    regs[2] |= (0u << 29);
    regs[2] |= (6u << 26);
    regs[2] |= (0u << 25);
    regs[2] |= (0u << 24);
    regs[2] |= ((r_counter & 0x03FFu) << 14);
    regs[2] |= (0u << 13);
    regs[2] |= (7u << 9);
    regs[2] |= ((ldf & 0x1u) << 8);
    regs[2] |= ((ldp & 0x1u) << 7);
    regs[2] |= (1u << 6);
    regs[2] |= (0u << 5);
    regs[2] |= (0u << 4);
    regs[2] |= (0u << 3);
    regs[2] |= 2u;

    regs[3] = 0u;
    regs[3] |= (0u << 23);
    regs[3] |= (0u << 22);
    regs[3] |= (0u << 21);
    regs[3] |= (0u << 18);
    regs[3] |= (0u << 15);
    regs[3] |= ((150u & 0x0FFFu) << 3);
    regs[3] |= 3u;

    regs[4] = 0u;
    regs[4] |= (1u << 23);
    regs[4] |= ((rf_div_sel & 0x7u) << 20);
    regs[4] |= ((band_sel_clkdiv & 0xFFu) << 12);
    regs[4] |= (0u << 11);
    regs[4] |= (0u << 10);
    regs[4] |= (0u << 9);
    regs[4] |= (0u << 8);
    regs[4] |= (0u << 6);
    regs[4] |= (1u << 5);
    regs[4] |= (3u << 3);
    regs[4] |= 4u;

    regs[5] = 0x00580005u;

    return 0;
}

static int adf4351_set_frequency(uint64_t fout_hz) {
    int rc = adf4351_build_regs(fout_hz, F_REF_HZ, DEFAULT_R_COUNTER, g_adf_regs);
    if (rc != 0) {
        return rc;
    }

    adf_program_regs(g_adf_regs);
    g_adf_regs_valid = true;
    return 0;
}

static int adf4351_set_output_enabled(bool enabled) {
    if (!g_adf_regs_valid) {
        return -1;
    }

    if (enabled) {
        g_adf_regs[4] |= (1u << 5);
    } else {
        g_adf_regs[4] &= ~(1u << 5);
    }

    adf_write32(g_adf_regs[4]);
    return 0;
}

// ============================================================
// TSL2591 helpers
// ============================================================
static inline uint8_t reg_cmd(uint8_t reg) {
    return (uint8_t)(TSL2591_CMD | reg);
}

static bool i2c_write_reg_u8(uint8_t reg, uint8_t val) {
    uint8_t buf[2] = { reg_cmd(reg), val };
    int rc = i2c_write_blocking(I2C_PORT, TSL2591_ADDR, buf, 2, false);
    return (rc == 2);
}

static bool i2c_read_reg(uint8_t reg, uint8_t *dst, size_t n) {
    uint8_t r = reg_cmd(reg);
    int rc1 = i2c_write_blocking(I2C_PORT, TSL2591_ADDR, &r, 1, true);
    if (rc1 != 1) return false;
    int rc2 = i2c_read_blocking(I2C_PORT, TSL2591_ADDR, dst, n, false);
    return (rc2 == (int)n);
}

static bool i2c_read_reg_u8(uint8_t reg, uint8_t *val) {
    return i2c_read_reg(reg, val, 1);
}

static bool i2c_read_reg_u16_le(uint8_t reg_low, uint16_t *val) {
    uint8_t b[2];
    if (!i2c_read_reg(reg_low, b, 2)) return false;
    *val = (uint16_t)(b[0] | (b[1] << 8));
    return true;
}

static bool tsl2591_read_id(uint8_t *id) {
    return i2c_read_reg_u8(REG_ID, id);
}

static bool tsl2591_enable(bool enable_als) {
    if (!i2c_write_reg_u8(REG_ENABLE, ENABLE_PON)) return false;
    sleep_ms(3);
    if (enable_als) {
        if (!i2c_write_reg_u8(REG_ENABLE, (uint8_t)(ENABLE_PON | ENABLE_AEN))) return false;
    }
    return true;
}

static bool tsl2591_disable(void) {
    return i2c_write_reg_u8(REG_ENABLE, 0x00);
}

static bool tsl2591_set_control(tsl2591_gain_t gain, tsl2591_it_t it) {
    uint8_t ctrl = (uint8_t)(gain | (uint8_t)it);
    return i2c_write_reg_u8(REG_CONTROL, ctrl);
}

static uint32_t tsl2591_integration_ms(tsl2591_it_t it) {
    return 100u * ((uint32_t)it + 1u);
}

static bool tsl2591_read_channels(uint16_t *ch0, uint16_t *ch1) {
    if (!i2c_read_reg_u16_le(REG_CHAN0_LOW, ch0)) return false;
    if (!i2c_read_reg_u16_le(REG_CHAN1_LOW, ch1)) return false;
    return true;
}

static float tsl2591_calc_lux(uint16_t ch0, uint16_t ch1, tsl2591_gain_t gain, tsl2591_it_t it) {
    if (ch0 == 0) return 0.0f;
    if (ch0 == 0xFFFFu || ch1 == 0xFFFFu) return -1.0f;

    float gain_scale = 1.0f;
    switch (gain) {
        case TSL2591_GAIN_LOW:  gain_scale = 1.0f; break;
        case TSL2591_GAIN_MED:  gain_scale = 25.0f; break;
        case TSL2591_GAIN_HIGH: gain_scale = 428.0f; break;
        case TSL2591_GAIN_MAX:  gain_scale = 9876.0f; break;
        default: gain_scale = 1.0f; break;
    }

    float atime_ms = (float)tsl2591_integration_ms(it);
    float cpl = (atime_ms * gain_scale) / 408.0f;
    if (cpl <= 0.0f) return 0.0f;

    float f_ch0 = (float)ch0;
    float f_ch1 = (float)ch1;
    float ratio = f_ch1 / f_ch0;

    float lux = ((f_ch0 - f_ch1) * (1.0f - ratio)) / cpl;
    if (lux < 0.0f) lux = 0.0f;
    return lux;
}

static bool tsl2591_init_sensor(void) {
    uint8_t id = 0;

    i2c_init(I2C_PORT, I2C_BAUD_HZ);
    gpio_set_function(I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_PIN);
    gpio_pull_up(I2C_SCL_PIN);

    sleep_ms(10);

    if (!tsl2591_read_id(&id)) {
        printf("ERR,read_id_failed\r\n");
        return false;
    }

    printf("ID,0x%02X\r\n", id);

    if (!tsl2591_disable()) {
        printf("WARN,disable_failed\r\n");
    }
    if (!tsl2591_set_control(g_sensor_gain, g_sensor_it)) {
        printf("ERR,set_control_failed\r\n");
        return false;
    }
    if (!tsl2591_enable(true)) {
        printf("ERR,enable_failed\r\n");
        return false;
    }

    printf("CFG,gain=0x%02X,it_ms=%lu\r\n",
           (unsigned)g_sensor_gain,
           (unsigned long)tsl2591_integration_ms(g_sensor_it));

    return true;
}

static bool tsl2591_take_sample(float *metric_out, uint16_t *ch0_out, uint16_t *ch1_out, float *lux_out) {
    uint16_t ch0 = 0;
    uint16_t ch1 = 0;
    float lux;

    sleep_ms(tsl2591_integration_ms(g_sensor_it) + g_sensor_extra_wait_ms);

    if (!tsl2591_read_channels(&ch0, &ch1)) {
        return false;
    }

    lux = tsl2591_calc_lux(ch0, ch1, g_sensor_gain, g_sensor_it);

    if (ch0_out) *ch0_out = ch0;
    if (ch1_out) *ch1_out = ch1;
    if (lux_out) *lux_out = lux;

#if USE_CH0_METRIC
    if (metric_out) *metric_out = (float)ch0;
#else
    if (metric_out) *metric_out = (lux < 0.0f) ? (float)ch0 : lux;
#endif

    return true;
}
static bool tsl2591_take_averaged_sample(
    float *metric_mean_out,
    float *metric_std_out,
    uint32_t *n_out
) {
    float sum = 0.0f;
    float sumsq = 0.0f;
    uint32_t i;

    for (i = 0; i < g_meas_settle_discard_samples; i++) {
        float metric_dummy = 0.0f;
        uint16_t ch0_dummy = 0;
        uint16_t ch1_dummy = 0;
        float lux_dummy = 0.0f;

        if (!tsl2591_take_sample(&metric_dummy, &ch0_dummy, &ch1_dummy, &lux_dummy)) {
            return false;
        }
    }

    for (i = 0; i < g_meas_avg_samples_per_state; i++) {
        float metric = 0.0f;
        uint16_t ch0 = 0;
        uint16_t ch1 = 0;
        float lux = 0.0f;

        if (!tsl2591_take_sample(&metric, &ch0, &ch1, &lux)) {
            return false;
        }

        sum += metric;
        sumsq += metric * metric;
    }

    if (n_out) {
        *n_out = g_meas_avg_samples_per_state;
    }

    if (metric_mean_out) {
        *metric_mean_out = sum / (float)g_meas_avg_samples_per_state;
    }

    if (metric_std_out) {
        float n = (float)g_meas_avg_samples_per_state;
        float mean = sum / n;
        float var = (sumsq / n) - (mean * mean);
        if (var < 0.0f) {
            var = 0.0f;
        }
        *metric_std_out = sqrtf(var);
    }

    return true;
}

static bool stream_one_sensor_line(void) {
    uint16_t ch0 = 0, ch1 = 0;
    float lux = 0.0f;
    uint32_t ms;

    if (!tsl2591_take_sample(NULL, &ch0, &ch1, &lux)) {
        printf("ERR,read_channels_failed\r\n");
        return false;
    }

    ms = to_ms_since_boot(get_absolute_time());
    gpio_xor_mask(1u << PICO_DEFAULT_LED_PIN);

    if (lux < 0.0f) {
        printf("DATA,%lu,%u,%u,\r\n", (unsigned long)ms, ch0, ch1);
    } else {
        printf("DATA,%lu,%u,%u,%.3f\r\n", (unsigned long)ms, ch0, ch1, lux);
    }

    return true;
}

// ============================================================
// Measurement routine
// ============================================================
static int perform_measurement(uint64_t freq_hz) {
    float off1 = 0.0f, on = 0.0f, off2 = 0.0f;
    float off1_std = 0.0f, on_std = 0.0f, off2_std = 0.0f;
    uint32_t off1_n = 0, on_n = 0, off2_n = 0;
    float off_ref, delta, contrast, drift;
    int rc;

    rc = adf4351_set_frequency(freq_hz);
    if (rc != 0) {
        printf("ERR,setfreq,%d\r\n", rc);
        return rc;
    }

    sleep_ms(g_freq_settle_ms);

    rc = adf4351_set_output_enabled(false);
    if (rc != 0) {
        printf("ERR,rf_off1,%d\r\n", rc);
        return rc;
    }
    sleep_ms(g_rf_off_settle_ms);

    if (!tsl2591_take_averaged_sample(&off1, &off1_std, &off1_n)) {
        printf("ERR,sample_off1\r\n");
        return -20;
    }

    rc = adf4351_set_output_enabled(true);
    if (rc != 0) {
        printf("ERR,rf_on,%d\r\n", rc);
        return rc;
    }
    sleep_ms(g_rf_on_settle_ms);

    if (!tsl2591_take_averaged_sample(&on, &on_std, &on_n)) {
        printf("ERR,sample_on\r\n");
        return -21;
    }

    rc = adf4351_set_output_enabled(false);
    if (rc != 0) {
        printf("ERR,rf_off2,%d\r\n", rc);
        return rc;
    }
    sleep_ms(g_rf_off_settle_ms);

    if (!tsl2591_take_averaged_sample(&off2, &off2_std, &off2_n)) {
        printf("ERR,sample_off2\r\n");
        return -22;
    }

    off_ref = 0.5f * (off1 + off2);
    delta = on - off_ref;
    drift = off2 - off1;
    contrast = (off_ref != 0.0f) ? (delta / off_ref) : 0.0f;

    printf("RESULT,%llu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\r\n",
           freq_hz, off1, on, off2, off_ref, delta, contrast, drift);

    printf("DBG,%llu,OFF1_STD=%.6f,ON_STD=%.6f,OFF2_STD=%.6f,N=%lu\r\n",
           freq_hz, off1_std, on_std, off2_std, (unsigned long)off1_n);

    return 0;
}
// ============================================================
// Command handler
// ============================================================
static void handle_command(const char *line) {
    char buf[96];
    char lower[96];
    char *arg;
    uint64_t freq_hz;
    uint32_t u32val;
    tsl2591_it_t new_it;
    int rc;

    strncpy(buf, line, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    trim_inplace(buf);

    if (buf[0] == '\0') {
        return;
    }

    strncpy(lower, buf, sizeof(lower) - 1);
    lower[sizeof(lower) - 1] = '\0';
    str_to_lower_inplace(lower);

    if (strcmp(lower, "mode odmr") == 0) {
        g_mode = RUN_MODE_ODMR;
        g_stream_enabled = false;
        adf4351_set_output_enabled(false);
        printf("OK,MODE,ODMR\r\n");
        return;
    }

    if (strcmp(lower, "mode stream") == 0) {
        g_mode = RUN_MODE_STREAM;
        g_stream_enabled = true;
        adf4351_set_output_enabled(false);
        printf("OK,MODE,STREAM\r\n");
        return;
    }

    if (strcmp(lower, "stream on") == 0) {
        g_mode = RUN_MODE_STREAM;
        g_stream_enabled = true;
        adf4351_set_output_enabled(false);
        printf("OK,STREAM,ON\r\n");
        return;
    }

    if (strcmp(lower, "stream off") == 0) {
        g_stream_enabled = false;
        printf("OK,STREAM,OFF\r\n");
        return;
    }

    if (strcmp(lower, "getcfg") == 0) {
        print_config();
        return;
    }

    if (strncmp(lower, "cfg it_ms ", 10) == 0) {
        arg = buf + 10;
        trim_inplace(arg);

        if (!parse_u32_arg(arg, &u32val)) {
            printf("ERR,cfg,it_ms,parse\r\n");
            return;
        }

        if (!tsl2591_it_from_ms(u32val, &new_it)) {
            printf("ERR,cfg,it_ms,allowed=100|200|300|400|500|600\r\n");
            return;
        }

        g_sensor_it = new_it;
        if (!apply_sensor_control_runtime()) {
            printf("ERR,cfg,it_ms,apply\r\n");
            return;
        }

        printf("OK,CFG,IT_MS,%lu\r\n", (unsigned long)u32val);
        return;
    }

    if (strncmp(lower, "cfg extra_wait_ms ", 18) == 0) {
        arg = buf + 18;
        trim_inplace(arg);

        if (!parse_u32_arg(arg, &u32val)) {
            printf("ERR,cfg,extra_wait_ms,parse\r\n");
            return;
        }

        g_sensor_extra_wait_ms = u32val;
        printf("OK,CFG,EXTRA_WAIT_MS,%lu\r\n", (unsigned long)u32val);
        return;
    }

    if (strncmp(lower, "cfg freq_settle_ms ", 19) == 0) {
        arg = buf + 19;
        trim_inplace(arg);

        if (!parse_u32_arg(arg, &u32val)) {
            printf("ERR,cfg,freq_settle_ms,parse\r\n");
            return;
        }

        g_freq_settle_ms = u32val;
        printf("OK,CFG,FREQ_SETTLE_MS,%lu\r\n", (unsigned long)u32val);
        return;
    }

    if (strncmp(lower, "cfg rf_on_settle_ms ", 20) == 0) {
        arg = buf + 20;
        trim_inplace(arg);

        if (!parse_u32_arg(arg, &u32val)) {
            printf("ERR,cfg,rf_on_settle_ms,parse\r\n");
            return;
        }

        g_rf_on_settle_ms = u32val;
        printf("OK,CFG,RF_ON_SETTLE_MS,%lu\r\n", (unsigned long)u32val);
        return;
    }

    if (strncmp(lower, "cfg rf_off_settle_ms ", 21) == 0) {
        arg = buf + 21;
        trim_inplace(arg);

        if (!parse_u32_arg(arg, &u32val)) {
            printf("ERR,cfg,rf_off_settle_ms,parse\r\n");
            return;
        }

        g_rf_off_settle_ms = u32val;
        printf("OK,CFG,RF_OFF_SETTLE_MS,%lu\r\n", (unsigned long)u32val);
        return;
    }

    if (strncmp(lower, "cfg discard ", 12) == 0) {
        arg = buf + 12;
        trim_inplace(arg);

        if (!parse_u32_arg(arg, &u32val)) {
            printf("ERR,cfg,discard,parse\r\n");
            return;
        }

        g_meas_settle_discard_samples = u32val;
        printf("OK,CFG,DISCARD,%lu\r\n", (unsigned long)u32val);
        return;
    }

    if (strncmp(lower, "cfg avg ", 8) == 0) {
        arg = buf + 8;
        trim_inplace(arg);

        if (!parse_u32_arg(arg, &u32val)) {
            printf("ERR,cfg,avg,parse\r\n");
            return;
        }

        if (u32val == 0u) {
            printf("ERR,cfg,avg,must_be_positive\r\n");
            return;
        }

        g_meas_avg_samples_per_state = u32val;
        printf("OK,CFG,AVG,%lu\r\n", (unsigned long)u32val);
        return;
    }

    if (strncmp(lower, "set ", 4) == 0) {
        arg = buf + 4;
        trim_inplace(arg);

        rc = parse_frequency_hz(arg, &freq_hz);
        if (rc != 0) {
            printf("ERR parse\r\n");
            return;
        }

        rc = adf4351_set_frequency(freq_hz);
        if (rc != 0) {
            printf("ERR set %d\r\n", rc);
            return;
        }

        adf4351_set_output_enabled(true);
        printf("OK %llu Hz\r\n", freq_hz);
        return;
    }

    if (strncmp(lower, "meas ", 5) == 0) {
        arg = buf + 5;
        trim_inplace(arg);

        rc = parse_frequency_hz(arg, &freq_hz);
        if (rc != 0) {
            printf("ERR parse\r\n");
            return;
        }

        g_mode = RUN_MODE_ODMR;
        g_stream_enabled = false;
        perform_measurement(freq_hz);
        return;
    }

    rc = parse_frequency_hz(buf, &freq_hz);
    if (rc == 0) {
        rc = adf4351_set_frequency(freq_hz);
        if (rc != 0) {
            printf("ERR set %d\r\n", rc);
            return;
        }

        adf4351_set_output_enabled(true);
        printf("OK %llu Hz\r\n", freq_hz);
        return;
    }

    printf("ERR unknown\r\n");
    printf("Commands: SET <freq>, MEAS <freq>, MODE ODMR, MODE STREAM, STREAM ON, STREAM OFF,\r\n");
    printf("          GETCFG,\r\n");
    printf("          CFG IT_MS <100|200|300|400|500|600>,\r\n");
    printf("          CFG EXTRA_WAIT_MS <ms>, CFG FREQ_SETTLE_MS <ms>,\r\n");
    printf("          CFG RF_ON_SETTLE_MS <ms>, CFG RF_OFF_SETTLE_MS <ms>,\r\n");
    printf("          CFG DISCARD <n>, CFG AVG <n>\r\n");
}
// ============================================================
// Main
// ============================================================
int main(void) {
    char line[128];
    size_t idx = 0;

    led_init();
    led_blink(3, 100, 100);

    stdio_init_all();
    sleep_ms(2000);

    printf("BOOT,main_entered\r\n");

    spi_init(SPI_PORT, SPI_BAUD);
    spi_set_format(SPI_PORT, 8, SPI_CPOL_0, SPI_CPHA_0, SPI_MSB_FIRST);

    gpio_set_function(PIN_SPI_SCK, GPIO_FUNC_SPI);
    gpio_set_function(PIN_SPI_MOSI, GPIO_FUNC_SPI);

    gpio_init(PIN_LE);
    gpio_set_dir(PIN_LE, GPIO_OUT);
    gpio_put(PIN_LE, 0);

    gpio_init(PIN_CE);
    gpio_set_dir(PIN_CE, GPIO_OUT);
    gpio_put(PIN_CE, 1);

    sleep_ms(10);

    if (adf4351_set_frequency(2870000000ULL) == 0) {
        adf4351_set_output_enabled(false);
        printf("ADF4351 ready. Default 2870000000 Hz\r\n");
    } else {
        printf("ADF4351 init failed\r\n");
    }

    if (!tsl2591_init_sensor()) {
        printf("ERR,sensor_init_failed\r\n");
        while (1) {
            gpio_xor_mask(1u << PICO_DEFAULT_LED_PIN);
            sleep_ms(500);
        }
    }

printf("READY,commands=SET|MEAS|MODE ODMR|MODE STREAM|STREAM ON|STREAM OFF|GETCFG|CFG ...\r\n");
print_config();
    while (1) {
        int ch = getchar_timeout_us(0);

        if (ch != PICO_ERROR_TIMEOUT) {
            if (ch == '\r' || ch == '\n') {
                if (idx > 0) {
                    line[idx] = '\0';
                    handle_command(line);
                    idx = 0;
                }
            } else if (idx < sizeof(line) - 1) {
                line[idx++] = (char)ch;
            } else {
                idx = 0;
                printf("ERR line too long\r\n");
            }
        }

        if (g_mode == RUN_MODE_STREAM && g_stream_enabled) {
            stream_one_sensor_line();
        } else {
            tight_loop_contents();
        }
    }
}
