#pragma once
// ============================================================================
//  Predictive fan controller (tiny neural network, runs on the ESP32-S3)
//
//  Once a minute push() receives CO2, dust, sound, light and the fan speed. A 12-16-2 network forecasts how
//  much CO2 and dust will change over the next 15 minutes for a given fan speed. The controller asks it about
//  5 fan speeds (CAND) and takes the LOWEST speed whose forecast stays under the CO2 and dust limits.
//  The choice is made with sigmoid weights, so there is no if / else / ?: / switch anywhere in this file:
//
//     ok_i   = sigmoid((co2Limit - co2_forecast_i)/SOFT_CO2) * sigmoid((dustLimit - dust_forecast_i)/SOFT_DUST)
//     weight = ok_i * (1-ok_0) * ... * (1-ok_{i-1})          "candidate i is the lowest one that is enough"
//     fan    = sum(weight_i * CAND_i) + (1 - sum(weights)) * 100      "nothing is enough -> full speed"
//
//  A slow bias term follows the forecast error (actual change - predicted change, 15 min later), so the model
//  adapts to the real room. Weights: include/fan_ai_weights.h (tools/train_fan_ai.py).
//  Keep the constants identical to tools/train_fan_ai.py.
// ============================================================================
#include <math.h>
#include <stdint.h>
#include <algorithm>
#include "fan_ai_weights.h"

namespace fanai {

constexpr int N_IN = 12;
constexpr int N_HID = 16;
constexpr int HIST = 32;  // ring buffer length (power of two): one sample per minute
constexpr int HORIZON = 15;
constexpr int N_CAND = 5;
static const float CAND[N_CAND] = {0.0f, 30.0f, 50.0f, 75.0f, 100.0f};
constexpr float SOFT_CO2 = 60.0f;   // ppm   : how "soft" the CO2 limit is
constexpr float SOFT_DUST = 8.0f;   // ug/m3
constexpr float EMA = 0.3f;         // output smoothing per minute
constexpr float BIAS_LR = 0.02f;    // forecast-error adaptation per minute

class FanAI {
 public:
  void setLimits(float co2Ppm, float dustUgm3) {
    co2Lim_ = co2Ppm;
    dustLim_ = dustUgm3;
  }

  // Call once per minute. fanPct = fan speed actually applied (0-100).
  void push(float co2, float dust, float sound, float lux, float fanPct) {
    const uint32_t i = n_ & MASK;
    co2_[i] = co2;
    dust_[i] = dust;
    snd_[i] = sound;
    lux_[i] = lux;
    fan_[i] = fanPct;
    n_++;

    // forecast at the current fan speed (kept for the error feedback and for telemetry)
    float x[N_IN];
    features(x, fanPct);
    float dC, dD;
    forward(x, dC, dD);
    rawC_[i] = dC;
    rawD_[i] = dD;

    // learn from the forecast made HORIZON minutes ago ("valid" is 0.0 until that sample exists)
    const uint32_t k = (n_ - 1u - HORIZON) & MASK;
    const float valid = (float)(n_ >= (uint32_t)HORIZON + 1u);
    const float errC = (co2 - co2_[k]) - rawC_[k];
    const float errD = (dust - dust_[k]) - rawD_[k];
    biasC_ = fminf(400.0f, fmaxf(-400.0f, biasC_ + BIAS_LR * valid * (errC - biasC_)));
    biasD_ = fminf(60.0f, fmaxf(-60.0f, biasD_ + BIAS_LR * valid * (errD - biasD_)));
    predCo2_ = co2 + dC + biasC_;
    predDust_ = dust + dD + biasD_;

    // soft "lowest sufficient fan speed"
    float rest = 1.0f, f = 0.0f;
    for (int c = 0; c < N_CAND; c++) {
      x[10] = CAND[c] / 100.0f;
      float pc, pd;
      forward(x, pc, pd);
      const float ok = sigmoid((co2Lim_ - (co2 + pc + biasC_)) / SOFT_CO2) *
                       sigmoid((dustLim_ - (dust + pd + biasD_)) / SOFT_DUST);
      f += rest * ok * CAND[c];
      rest *= (1.0f - ok);
    }
    f += rest * 100.0f;
    out_ += EMA * (f - out_);
  }

  float fan() const { return out_; }              // recommended fan speed, 0-100 %
  float predCo2() const { return predCo2_; }      // forecast CO2 in 15 min at the current fan speed
  float predDust() const { return predDust_; }
  float biasCo2() const { return biasC_; }
  uint32_t samples() const { return n_; }

 private:
  static constexpr uint32_t MASK = HIST - 1;

  static float sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

  // value `lag` minutes ago (lag is clamped to the history that exists)
  float at(const float* b, int lag) const {
    const uint32_t l = std::min<uint32_t>((uint32_t)lag, n_ - 1u);
    return b[(n_ - 1u - l) & MASK];
  }

  void features(float* x, float candFan) const {
    const float c = at(co2_, 0), d = at(dust_, 0);
    float fsum = 0.0f;
    for (int l = 0; l < HORIZON; l++) fsum += at(fan_, l);
    x[0] = c / 1000.0f;
    x[1] = (c - at(co2_, 5)) / 200.0f;
    x[2] = (c - at(co2_, 15)) / 400.0f;
    x[3] = (c - at(co2_, 30)) / 600.0f;
    x[4] = d / 100.0f;
    x[5] = (d - at(dust_, 5)) / 40.0f;
    x[6] = (d - at(dust_, 15)) / 80.0f;
    x[7] = (d - at(dust_, 30)) / 120.0f;
    x[8] = at(snd_, 0) / 100.0f;
    x[9] = at(lux_, 0) / 1000.0f;
    x[10] = candFan / 100.0f;
    x[11] = fsum / (float)HORIZON / 100.0f;
  }

  // network output: change of CO2 (ppm) and dust (ug/m3) over the next HORIZON minutes
  static void forward(const float* x, float& dCo2, float& dDust) {
    float h[N_HID];
    for (int j = 0; j < N_HID; j++) {
      float s = B1[j];
      for (int i = 0; i < N_IN; i++) s += W1[j][i] * x[i];
      h[j] = tanhf(s);
    }
    float o0 = B2[0], o1 = B2[1];
    for (int j = 0; j < N_HID; j++) {
      o0 += W2[0][j] * h[j];
      o1 += W2[1][j] * h[j];
    }
    dCo2 = o0 * 1000.0f;
    dDust = o1 * 100.0f;
  }

  float co2_[HIST] = {0}, dust_[HIST] = {0}, snd_[HIST] = {0}, lux_[HIST] = {0}, fan_[HIST] = {0};
  float rawC_[HIST] = {0}, rawD_[HIST] = {0};
  uint32_t n_ = 0;
  float biasC_ = 0, biasD_ = 0, out_ = 0, predCo2_ = 0, predDust_ = 0;
  float co2Lim_ = 1000.0f, dustLim_ = 60.0f;
};

}  // namespace fanai
