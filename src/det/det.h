#pragma once

#include <hps/src/hps.h>
#include "half_det.h"

class Det {
 public:
  HalfDet up;

  HalfDet dn;

  Det(){};

  Det(const size_t n_up_hf, const size_t n_dn_hf) : up(HalfDet(n_up_hf)), dn(HalfDet(n_dn_hf)) {}
};

namespace hps {
template <class B>
class Serializer<Det, B> {
 public:
  static void serialize(const Det& det, OutputBuffer<B>& buf) {
    Serializer<HalfDet, B>::serialize(det.up, buf);
    Serializer<HalfDet, B>::serialize(det.dn, buf);
  }
  static void parse(Det& det, InputBuffer<B>& buf) {
    Serializer<HalfDet, B>::parse(det.up, buf);
    Serializer<HalfDet, B>::parse(det.dn, buf);
  }
};
}  // namespace hps